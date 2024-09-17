import math
import multiprocessing
import os
import pickle
import sys
import concurrent.futures
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from fun_search.refined_algorithm import boxes, H, W, L, main_entry
import gradio as gr


class PackingEnvironment:
    def __init__(self, L, W, H, boxes):
        self.L = L
        self.W = W
        self.H = H
        self.boxes = boxes
        self.boxes.sort(key=lambda x: x[0] * x[1] * x[2], reverse=True)

        self.current_order = list(range(len(boxes)))
        self.state_dim = len(boxes)
        self.action_dim = 2

    def reset(self):
        self.current_order = list(range(len(self.boxes)))
        return self.get_state()

    def get_state(self):
        # Return the current order and other relevant state information
        return self.current_order

    def swap(self, i, j):
        self.current_order[i], self.current_order[j] = self.current_order[j], self.current_order[i]

    def step(self, action):
        # Action is a tuple (i, j) representing a swap
        self.swap(*action)
        _, placed_boxes, _ = main_entry([self.boxes[idx] for idx in self.current_order])
        # Calculate the fill-in rate as reward
        return self.get_state(), sum(l * w * h for l, w, h, _, _, _ in placed_boxes) / (self.L * self.W * self.H)


class PackingDataset(Dataset):
    def __init__(self, env_, num_samples, dataset_file='packing_dataset.pkl', force_reload_data=False):
        self.env = env_
        self.num_samples = num_samples
        self.dataset_file = dataset_file

        if os.path.exists(self.dataset_file) and not force_reload_data:
            tqdm.write("Dataset exists already")
            self.data = self.load_data()
        else:
            tqdm.write("Generate dataset since not exists")
            self.data = self.generate_data_multi_thread()
            self.save_data()

    def generate_single_sample(self, env_, max_steps):
        state = env_.reset()
        trajectory = []
        time_steps = []
        traj_mask = np.zeros(max_steps, dtype=int)

        for step in range(max_steps):
            i, j = np.random.choice(len(state), 2, replace=False)
            action = (i, j)
            traj_mask[step] = 1
            next_state, reward = env_.step(action)
            trajectory.append((state, action, reward, traj_mask.copy()))
            time_steps.append(step)
            state = next_state

        return trajectory, time_steps

    def generate_data(self, progress=gr.Progress()):
        data = []
        max_steps = int(len(self.env.boxes) / 2)

        for i in range(self.num_samples):
            trajectory, time_steps = self.generate_single_sample(self.env, max_steps)
            data.append((trajectory, time_steps))
            progress((i * max_steps, self.num_samples * max_steps), desc="Preparing data")
        return data

    def generate_data_multi_thread(self, progress=gr.Progress()):
        data = []
        max_steps = int(len(self.env.boxes) / 2)
        finish_steps = 0

        # Multiprocessing pool
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [executor.submit(self.generate_single_sample, self.env, max_steps)
                       for _ in range(self.num_samples)]

            for future in concurrent.futures.as_completed(futures):
                data.append(future.result())
                finish_steps += max_steps
                # Update the progress bar here in the main process
                progress((finish_steps, self.num_samples * max_steps), desc="Preparing data")
        return data

    def save_data(self):
        with open(self.dataset_file, 'wb') as f:
            pickle.dump(self.data, f)
        tqdm.write(f"Dataset saved to {self.dataset_file}")

    def load_data(self):
        with open(self.dataset_file, 'rb') as f:
            data = pickle.load(f)
        tqdm.write(f"Dataset loaded from {self.dataset_file}")
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        trajectory, time_steps = self.data[idx]
        states, actions, rewards, traj_mask = zip(*trajectory)
        returns_to_go = np.cumsum(rewards[::-1])[::-1]
        return torch.from_numpy(np.asarray(time_steps)), torch.from_numpy(np.asarray(states)).float(), \
            torch.from_numpy(np.asarray(actions)).float(), torch.from_numpy(returns_to_go.copy()).float(), \
            torch.from_numpy(np.asarray(traj_mask))


class MaskedCausalAttention(nn.Module):
    def __init__(self, h_dim, max_T, n_heads, drop_p):
        super().__init__()

        self.n_heads = n_heads
        self.max_T = max_T

        self.q_net = nn.Linear(h_dim, h_dim)
        self.k_net = nn.Linear(h_dim, h_dim)
        self.v_net = nn.Linear(h_dim, h_dim)

        self.proj_net = nn.Linear(h_dim, h_dim)

        self.att_drop = nn.Dropout(drop_p)
        self.proj_drop = nn.Dropout(drop_p)

        ones = torch.ones((max_T, max_T))
        mask = torch.tril(ones).view(1, 1, max_T, max_T)

        # register buffer makes sure mask does not get updated during backpropagation
        self.register_buffer('mask', mask)

    def forward(self, x):
        B, T, _, h_dim = x.shape  # batch size, seq length, h_dim * n_heads
        C = self.n_heads * h_dim
        N, D = self.n_heads, C // self.n_heads  # N = num heads, D = attention dim

        # rearrange q, k, v as (B, N, T, D)
        q = self.q_net(x).view(B, T, N, D).transpose(1, 2)
        k = self.k_net(x).view(B, T, N, D).transpose(1, 2)
        v = self.v_net(x).view(B, T, N, D).transpose(1, 2)

        # weights (B, N, T, T)
        weights = q @ k.transpose(2, 3) / math.sqrt(D)
        # causal mask applied to weights
        weights = weights.masked_fill(self.mask[..., :T, :T] == 0, float('-inf'))
        # normalize weights, all -inf -> 0 after softmax
        normalized_weights = F.softmax(weights, dim=-1)
        # attention (B, N, T, D)
        attention = self.att_drop(normalized_weights @ v)
        # gather heads and project (B, N, T, D) -> (B, T, N*D)
        attention = attention.transpose(1, 2).contiguous()
        return self.proj_drop(self.proj_net(attention))


class Block(nn.Module):
    def __init__(self, h_dim, max_T, n_heads, drop_p):
        super().__init__()
        self.attention = MaskedCausalAttention(h_dim, max_T, n_heads, drop_p)
        self.mlp = nn.Sequential(
            nn.Linear(h_dim, 4 * h_dim),
            nn.GELU(),
            nn.Linear(4 * h_dim, h_dim),
            nn.Dropout(drop_p),
        )
        self.ln1 = nn.LayerNorm(h_dim)
        self.ln2 = nn.LayerNorm(h_dim)

    def forward(self, x):
        # Attention -> LayerNorm -> MLP -> LayerNorm
        x = x + self.attention(x)  # residual
        x = self.ln1(x)
        x = x + self.mlp(x)  # residual
        x = self.ln2(x)
        return x


class DecisionTransformer(nn.Module):
    def __init__(self, state_dim, act_dim, n_blocks, h_dim, context_len, n_heads, drop_p, max_timestep=4096):
        super().__init__()

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.h_dim = h_dim

        # transformer blocks
        input_seq_len = 3 * context_len
        blocks = [Block(h_dim, input_seq_len, n_heads, drop_p) for _ in range(n_blocks)]
        self.transformer = nn.Sequential(*blocks)

        # projection heads (project to embedding)
        self.embed_ln = nn.LayerNorm(h_dim)
        self.embed_timestep = nn.Embedding(max_timestep, h_dim)
        self.embed_rtg = torch.nn.Linear(1, h_dim)
        self.embed_state = torch.nn.Linear(state_dim, h_dim)
        # discrete actions
        self.embed_action = torch.nn.Linear(act_dim, h_dim)

        # prediction heads
        self.predict_rtg = torch.nn.Linear(h_dim, 1)
        self.predict_state = torch.nn.Linear(h_dim, state_dim)
        self.predict_action = nn.Sequential(*([nn.Linear(h_dim, act_dim)]))

    def forward(self, time_steps, states, actions, returns_to_go):
        B, T, _ = states.shape
        time_embeddings = self.embed_timestep(time_steps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = self.embed_state(states) + time_embeddings
        returns_embeddings = self.embed_rtg(returns_to_go) + time_embeddings
        action_embeddings = self.embed_action(actions) + time_embeddings

        # stack rtg, states and actions and reshape sequence as
        # (r1, s1, a1, r2, s2, a2 ...)
        h = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3)
        # h = h.reshape(B, 3 * T, self.h_dim)
        h = self.embed_ln(h)
        # transformer and prediction
        h = self.transformer(h)

        # get h reshaped such that its size = (B x 3 x T x h_dim) and
        # h[:, 0, t] is conditioned on r_0, s_0, a_0 ... r_t
        # h[:, 1, t] is conditioned on r_0, s_0, a_0 ... r_t, s_t
        # h[:, 2, t] is conditioned on r_0, s_0, a_0 ... r_t, s_t, a_t
        # predict next state given r, s, a, next action given r, s, a and next rtg given r, s, a
        return self.predict_rtg(h[:, :, 0, :]), self.predict_state(h[:, :, 1, :]), self.predict_action(h[:, :, 2, :])


def train(max_train_iters, model, data_loader, device, optimizer, scheduler, progress=gr.Progress()):
    returns_to_go = None
    avg_loss = 0
    best_return = 0
    for i_train_iter in range(max_train_iters):
        action_losses = []
        model.train()
        for data in data_loader:
            time_steps, states, actions, returns_to_go, traj_mask = data
            time_steps = time_steps.to(device)  # B x T
            states = states.to(device)  # B x T x state_dim
            actions = actions.to(device)  # B x T x act_dim
            returns_to_go = returns_to_go.to(device).unsqueeze(dim=-1)  # B x T x 1
            action_target = torch.clone(actions).detach().to(device)

            _, _, action_preds = model.forward(
                time_steps=time_steps,
                states=states,
                actions=actions,
                returns_to_go=returns_to_go
            )
            # Only consider non-padded elements
            action_loss = F.mse_loss(action_preds, action_target, reduction='mean')
            optimizer.zero_grad()
            action_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optimizer.step()
            scheduler.step()
            action_losses.append(action_loss.detach().cpu().item())
            avg_loss = np.mean(action_losses)

        reward_start = returns_to_go[0, 1, :].item()
        # reward_end = returns_to_go[0, -1, :].item()
        best_return = reward_start if reward_start > best_return else best_return
        progress_desc = f"Training - (Avg loss={avg_loss:.2f}, Reward={reward_start:.2f})"
        progress((i_train_iter + 1, max_train_iters), desc=progress_desc)

        # Early stop
        if avg_loss <= 100:
            break

    torch.save(model.state_dict(), 'model.pth')


def find_best_path(target_return, model, env, device):
    eval_data_loader = DataLoader(
        PackingDataset(env, num_samples=500),
        batch_size=1,
        shuffle=False
    )

    # Initialize environment
    env_ = PackingEnvironment(L, W, H, boxes)
    max_steps = int(len(env_.boxes) / 2)
    time_steps, states, actions, returns_to_go, traj_mask = next(iter(eval_data_loader))

    # Modify the states to the ordered sequence
    ordered_states = torch.arange(0, len(env_.boxes), device=device).unsqueeze(0).repeat(max_steps, 1).unsqueeze(
        0).float()
    states = ordered_states

    time_steps = time_steps.to(device)
    states = states.to(device)
    actions = actions.to(device)
    returns_to_go = returns_to_go.to(device)

    # Set the initial return value
    returns_to_go[0, 0] = torch.asarray(target_return).to(device)
    best_state = None
    best_return = 0

    result_pack_seq = ""
    # Loop through time steps
    for t in range(max_steps):
        # Perform prediction
        return_preds, state_preds, action_preds = model.forward(
            time_steps=time_steps[:, :t + 1],
            states=states[:, :t + 1, :],
            actions=actions[:, :t + 1, :],
            returns_to_go=returns_to_go[:, t]
        )

        # Update tensors
        actions[:, t, :] = action_preds[:, -1, :]
        action_tuple = (abs(int(actions[:, t, 0])), abs(int(actions[:, t, 1])))

        new_state, new_return = env_.step(action_tuple)
        # calculate current return-to-go
        target_return -= new_return
        states[:, t, :] = torch.asarray(new_state)
        if t != max_steps - 1:
            returns_to_go[:, t + 1] = target_return
        result_pack_seq += f"{action_tuple}, {returns_to_go[:, t].item()}\n"

        if new_return > best_return:
            best_state, best_return = torch.asarray(new_state), new_return

    # Extract final state from the last time step
    # final_state = states[:, -1, :]
    final_state = best_state
    boxes.sort(key=lambda x: x[0] * x[1] * x[2], reverse=True)
    reordered_boxes = [boxes[i] for i in final_state.long().cpu().flatten().tolist()]
    figs_refined, placed_boxes_refined, _ = main_entry(reordered_boxes)

    return figs_refined, placed_boxes_refined, reordered_boxes, result_pack_seq


def main_entry_opt():
    rtg_scale = 1000  # scale to normalize returns to go
    max_eval_ep_len = 1000  # max len of one evaluation episode
    num_eval_ep = 10  # num of evaluation episodes per iteration
    num_data_samples = 400  # num of data samples, increase will help on finding better solution
    batch_size = 16  # training batch size
    lr = 1e-4  # learning rate
    wt_decay = 1e-4  # weight decay
    warmup_steps = 10000  # warmup steps for lr scheduler
    max_train_iters = 1500  # 2000
    num_updates_per_iter = 100  # total updates = max_train_iters x num_updates_per_iter
    context_length = 80  # K in decision transformer
    num_blocks = 3  # num of transformer blocks
    embed_dim = 128  # embedding (hidden) dim of transformer
    num_heads = 3  # num of transformer heads
    dropout_p = 0.1  # dropout probability

    # training and evaluation device
    device_name = 'cuda'
    device = torch.device(device_name)
    tqdm.write(f"device set to: {device}")

    env = PackingEnvironment(L, W, H, boxes)
    model = DecisionTransformer(
        state_dim=env.state_dim,
        act_dim=env.action_dim,
        n_blocks=num_blocks,
        h_dim=embed_dim,
        context_len=context_length,
        n_heads=num_heads,
        drop_p=dropout_p,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wt_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda steps: min((steps + 1) / warmup_steps, 1))
    data_loader = DataLoader(
        PackingDataset(env, num_samples=num_data_samples, force_reload_data=True),
        batch_size=batch_size,
        shuffle=True
    )

    # train procedure
    train(max_train_iters, model, data_loader, device, optimizer, scheduler)

    # eval procedure
    model.load_state_dict(torch.load('model.pth'))
    model.eval()
    target_return = 55
    figs_refined, placed_boxes_refined_list, reordered_boxes, result_pack_seq = find_best_path(target_return, model, env, device)

    # The original packing result
    boxes.sort(key=lambda x: x[0] * x[1] * x[2], reverse=True)
    figs_normal, placed_box_list, failed_box_list = main_entry(boxes)

    fill_rate = result_pack_seq + f"original fill-in rate: {sum(l * w * h for l, w, h, _, _, _ in placed_box_list) / (L * W * H)}\n" \
        f"refined fill-in rate: {sum(l * w * h for l, w, h, _, _, _ in placed_boxes_refined_list) / (L * W * H)}"

    reordered_boxes_code = f"refined_packing_order_boxes = {reordered_boxes}"

    return figs_normal + figs_refined, fill_rate, reordered_boxes_code
