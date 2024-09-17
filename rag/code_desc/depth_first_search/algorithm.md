### 1. 要求

物流公司在流通过程中，需要将打包完毕的箱子装入到一个货车的车厢中，为了提高物流效率，需要将车厢尽量填满，显然，车厢如果能被 $100\%$ 填满是最优的，但通常认为，车厢能够填满 $85\%$，认为装箱是优化的。

设车厢为长方形，其长宽高分别为 $L,W,H$；共有$n$个箱子，箱子也为长方形，第 $i$ 个箱子的长宽高为 $l_i$, $w_i$, $h_i$（ $n$ 个箱子的体积总和是要远远大于车厢的体积），做以下假设和要求：

1. 长方形的车厢共有 8 个角，并设靠近驾驶室并位于下端的一个角的坐标为 $(0,0,0)$ ，车厢共6个面，其中长的4个面，以及靠近驾驶室的面是封闭的，只有一个面是开着的，用于工人搬运箱子

2. 需要计算出每个箱子在车厢中的坐标，即每个箱子摆放后，其和车厢坐标为 $(0,0,0)$ 的角相对应的角在车厢中的坐标，并计算车厢的填充率。

### 2. 启发式块贪心，或深度优先扩展算法求解

采用启发式思想来对问题进行求解，算法流程如下：

1. 取出当前可行块。

    在块选择算法过程中，可以采用贪心算法，直接返回填充体积最大的块，由于可行块列表已经按照体积降序排列，实际上算法选择的块总是列表的第一个元素。

    在块选择算法过程中，也可以采用深度优先搜索算法扩展当前放置方案，算法输入为一个部分放置方案，深度限制和最大分支数。深度优先搜索补全算法从一个部分放置方案出发，递归的尝试可行块列表中的块，在到达深度限制的时候调用补全函数得到当前方案的评估值，并记录整个搜索过程找到的最优的评估值作为输入部分放置方案的评估。

2. 取出剩余空间堆栈中的栈顶剩余空间。
3. 尝试放置当前可行块至栈顶剩余空间中。

    块放置算法将块和栈顶空间结合成一个放置加入当前放置方案，移除栈顶空间，扣除已使用物品，然后切割未填充空间并加入剩余空间堆栈。

    块移除算法从当前部分放置方案中移除当前块所属的放置，恢复已使用物品，移除空间堆栈栈顶的三个切割出来的剩余空间，并将已使用剩余空间重新插入栈顶。
    
4. 评估当前的状态，并将此评估值作为被选块的面适应度。

    评估当前部分放置方案好坏的最直接的方法，是用某种方式补全它，并以最终结果的填充率作为当前状态的评估值。该“补全评估算法”实际上是“整体基本启发式算法”的简化版，区别在于每个装载阶段算法都选择可行块列表中体积最大的块进行放置。
    
    由于可行块列表已按体积降序排列，实际算法选择的块总是列表第一个元素。算法不改变输入的部分放置方案，只是把最终补全的结果记录在此状态的 $volume_{complete}$ 域作为该状态的评估值。

5. 从当前状态移除当前可行块。
6. 判断可行块列表是否遍历完成，若是则选取适应度最高的块作为结果，若否则返回步骤 1，再次执行。


### 3. 代码示例
```python
import copy
import numpy as np
from itertools import product
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from classes import *

# Constants
MIN_FILL_RATE = 0.9
MIN_AREA_RATE = 0.9
MAX_TIMES = 2
MAX_DEPTH = 3
MAX_BRANCH = 2

# The class of stack, boxes and containers are all assumed to be pre-defined

# Temporary best packing state
tmp_best_ps = None

# Common checks for block combination
def combine_common_check(combine: Block, container: Space, num_list):
    if any([
        combine.lx > container.lx,
        combine.ly > container.ly,
        combine.lz > container.lz,
        (np.array(combine.require_list) > np.array(num_list)).any(),
        combine.volume / (combine.lx * combine.ly * combine.lz) < MIN_FILL_RATE,
        (combine.ax * combine.ay) / (combine.lx * combine.ly) < MIN_AREA_RATE,
        combine.times > MAX_TIMES
    ]):
        return False
    return True

# Combine common attributes
def combine_common(a: Block, b: Block, combine: Block):
    combine.require_list = (np.array(a.require_list) + np.array(b.require_list)).tolist()
    combine.volume = a.volume + b.volume
    combine.children = [a, b]
    combine.times = max(a.times, b.times) + 1

# Generate simple blocks
def gen_simple_block(container, box_list, num_list):
    block_table = []
    for box in box_list:
        for nx in range(1, num_list[box.type] + 1):
            for ny in range(1, num_list[box.type] // nx + 1):
                for nz in range(1, num_list[box.type] // (nx * ny) + 1):
                    if all([
                        box.lx * nx <= container.lx,
                        box.ly * ny <= container.ly,
                        box.lz * nz <= container.lz
                    ]):
                        requires = np.full_like(num_list, 0)
                        requires[box.type] = nx * ny * nz
                        block = Block(box.lx * nx, box.ly * ny, box.lz * nz, requires)
                        block.ax, block.ay = box.lx * nx, box.ly * ny
                        block.volume = box.lx * nx * box.ly * ny * box.lz * nz
                        block.times = 0
                        block_table.append(block)
    return sorted(block_table, key=lambda x: x.volume, reverse=True)

# Generate complex blocks
def gen_complex_block(container, box_list, num_list):
    block_table = gen_simple_block(container, box_list, num_list)
    for times in range(MAX_TIMES):
        new_block_table = []
        for i in range(len(block_table)):
            a = block_table[i]
            for j in range(len(block_table)):
                if j == i:
                    continue
                b = block_table[j]
                if a.times == times or b.times == times:
                    c = Block(0, 0, 0)
                    if a.ax == a.lx and b.ax == b.lx and a.lz == b.lz:
                        c.direction = "x"
                        c.ax = a.ax + b.ax
                        c.ay = min(a.ay, b.ay)
                        c.lx = a.lx + b.lx
                        c.ly = max(a.ly, b.ly)
                        c.lz = a.lz
                        combine_common(a, b, c)
                        if combine_common_check(c, container, num_list):
                            new_block_table.append(c)
                    elif a.ay == a.ly and b.ay == b.ly and a.lz == b.lz:
                        c.direction = "y"
                        c.ax = min(a.ax, b.ax)
                        c.ay = a.ay + b.ay
                        c.lx = max(a.lx, b.lx)
                        c.ly = a.ly + b.ly
                        c.lz = a.lz
                        combine_common(a, b, c)
                        if combine_common_check(c, container, num_list):
                            new_block_table.append(c)
                    elif a.ax >= b.lx and a.ay >= b.ly:
                        c.direction = "z"
                        c.ax, c.ay = b.ax, b.ay
                        c.lx, c.ly = a.lx, a.ly
                        c.lz = a.lz + b.lz
                        combine_common(a, b, c)
                        if combine_common_check(c, container, num_list):
                            new_block_table.append(c)
        block_table += new_block_table
        block_table = list(set(block_table))
    return sorted(block_table, key=lambda x: x.volume, reverse=True)

# Generate feasible block list
def gen_block_list(space: Space, avail, block_table):
    return [
        block for block in block_table
        if all([
            (np.array(block.require_list) <= np.array(avail)).all(),
            block.lx <= space.lx,
            block.ly <= space.ly,
            block.lz <= space.lz
        ])
    ]

# Generate residual space
def gen_residual_space(space: Space, block: Block):
    rmx = space.lx - block.lx
    rmy = space.ly - block.ly
    rmz = space.lz - block.lz
    if rmx >= rmy:
        drs_x = Space(space.x + block.lx, space.y, space.z, rmx, space.ly, space.lz, space)
        drs_y = Space(space.x, space.y + block.ly, space.z, block.lx, rmy, space.lz, space)
        drs_z = Space(space.x, space.y, space.z + block.lz, block.ax, block.ay, rmz, None)
        return drs_z, drs_y, drs_x
    else:
        drs_x = Space(space.x + block.lx, space.y, space.z, rmx, block.ly, space.lz, space)
        drs_y = Space(space.x, space.y + block.ly, space.z, space.lx, rmy, space.lz, space)
        drs_z = Space(space.x, space.y, space.z + block.lz, block.ax, block.ay, rmz, None)
        return drs_z, drs_x, drs_y

# Space transfer
def transfer_space(space: Space, space_stack: Stack):
    if space_stack.size() <= 1:
        space_stack.pop()
        return None
    discard = space
    space_stack.pop()
    target = space_stack.top()
    if discard.origin and target.origin and discard.origin == target.origin:
        new_target = copy.deepcopy(target)
        if discard.lx == discard.origin.lx:
            new_target.ly = discard.origin.ly
        elif discard.ly == discard.origin.ly:
            new_target.lx = discard.origin.lx
        else:
            return None
        space_stack.pop()
        space_stack.push(new_target)
        return target
    return None

# Restore space transfer
def transfer_space_back(space: Space, space_stack: Stack, revert_space: Space):
    space_stack.pop()
    space_stack.push(revert_space)
    space_stack.push(space)

# Place block
def place_block(ps: PackingState, block: Block):
    space = ps.space_stack.pop()
    ps.avail_list = (np.array(ps.avail_list) - np.array(block.require_list)).tolist()
    place = Place(space, block)
    ps.plan_list.append(place)
    ps.volume += block.volume
    cuboid1, cuboid2, cuboid3 = gen_residual_space(space, block)
    ps.space_stack.push(cuboid1, cuboid2, cuboid3)
    return place

# Remove block
def remove_block(ps: PackingState, block: Block, place: Place, space: Space):
    ps.avail_list = (np.array(ps.avail_list) + np.array(block.require_list)).tolist()
    ps.plan_list.remove(place)
    ps.volume -= block.volume
    for _ in range(3):
        ps.space_stack.pop()
    ps.space_stack.push(space)

# Complete placement
def complete(ps: PackingState, block_table):
    tmp = copy.deepcopy(ps)
    while tmp.space_stack.not_empty():
        space = tmp.space_stack.top()
        block_list = gen_block_list(space, ps.avail_list, block_table)
        if block_list:
            place_block(tmp, block_list[0])
        else:
            transfer_space(space, tmp.space_stack)
    ps.volume_complete = tmp.volume

# Depth-first search with limit
def depth_first_search(ps: PackingState, depth, branch, block_table):
    global tmp_best_ps
    if depth != 0:
        space = ps.space_stack.top()
        block_list = gen_block_list(space, ps.avail_list, block_table)
        if block_list:
            for i in range(min(branch, len(block_list))):
                place = place_block(ps, block_list[i])
                depth_first_search(ps, depth - 1, branch, block_table)
                remove_block(ps, block_list[i], place, space)
        else:
            old_target = transfer_space(space, ps.space_stack)
            if old_target:
                depth_first_search(ps, depth, branch, block_table)
                transfer_space_back(space, ps.space_stack, old_target)
    else:
        complete(ps, block_table)
        if ps.volume_complete > tmp_best_ps.volume_complete:
            tmp_best_ps = copy.deepcopy(ps)

# Estimate block
def estimate(ps: PackingState, block_table, search_params):
    global tmp_best_ps
    tmp_best_ps = PackingState([], Stack(), [])
    depth_first_search(ps, MAX_DEPTH, MAX_BRANCH, block_table)
    return tmp_best_ps.volume_complete

# Find next block
def find_next_block(ps: PackingState, block_list, block_table, search_params):
    best_fitness = 0
    best_block = block_list[0]
    for block in block_list:
        space = ps.space_stack.top()
        place = place_block(ps, block)
        fitness = estimate(ps, block_table, search_params)
        remove_block(ps, block, place, space)
        if fitness > best_fitness:
            best_fitness = fitness
            best_block = block
    return best_block

# Build box positions recursively
def build_box_position(block, init_pos, box_list):
    if not block.children and block.times == 0:
        box_idx = next((i for i, x in enumerate(block.require_list) if x > 0), -1)
        if box_idx > -1:
            box = box_list[box_idx]
            nx, ny, nz = block.lx / box.lx, block.ly / box.ly, block.lz / box.lz
            x_list, y_list, z_list = np.arange(nx) * box.lx, np.arange(ny) * box.ly, np.arange(nz) * box.lz
            dimensions = (np.array(list(product(x_list, y_list, z_list))) + np.array(init_pos)).tolist()
            return sorted([d + [box.lx, box.ly, box.lz] for d in dimensions], key=lambda x: (x[0], x[1], x[2]))
        return []
    pos = []
    for child in block.children:
        pos.extend(build_box_position(child, init_pos, box_list))
        if block.direction == "x":
            init_pos = (init_pos[0] + child.lx, init_pos[1], init_pos[2])
        elif block.direction == "y":
            init_pos = (init_pos[0], init_pos[1] + child.ly, init_pos[2])
        elif block.direction == "z":
            init_pos = (init_pos[0], init_pos[1], init_pos[2] + child.lz)
    return pos

# Cuboid data
def cuboid_data2(o, size=(1, 1, 1)):
    X = np.array([[[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0]],
                  [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]],
                  [[1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]],
                  [[0, 0, 1], [0, 0, 0], [0, 1, 0], [0, 1, 1]],
                  [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]],
                  [[0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1]]]).astype(float)
    for i in range(3):
        X[:, :, i] *= size[i]
    X += np.array(o)
    return X

# Draw packing result
def draw_packing_result(problem: Problem, ps: PackingState):
    fig = plt.figure()
    ax1 = mplot3d.Axes3D(fig)
    plot_linear_cube(ax1, 0, 0, 0, problem.container.lx, problem.container.ly, problem.container.lz)
    for p in ps.plan_list:
        box_pos = build_box_position(p.block, (p.space.x, p.space.y, p.space.z), problem.box_list)
        positions, sizes = [], []
        colors = ["blue"] * len(box_pos)
        for bp in sorted(box_pos, key=lambda x: (x[0], x[1], x[2])):
            positions.append((bp[0], bp[1], bp[2]))
            sizes.append((bp[3], bp[4], bp[5]))
        pc = plotCubeAt2(positions, sizes, colors=colors, edgecolor="k")
        ax1.add_collection3d(pc)
    plt.title('Packing Result')
    plt.show()

# Basic heuristic algorithm
def basic_heuristic(is_complex, search_params, problem: Problem):
    block_table = gen_complex_block(problem.container, problem.box_list, problem.num_list) if is_complex else gen_simple_block(problem.container, problem.box_list, problem.num_list)
    ps = PackingState(avail_list=problem.num_list)
    ps.space_stack.push(problem.container)
    while ps.space_stack.size() > 0:
        space = ps.space_stack.top()
        block_list = gen_block_list(space, ps.avail_list, block_table)
        if block_list:
            block = find_next_block(ps, block_list, block_table, search_params)
            ps.space_stack.pop()
            ps.avail_list = (np.array(ps.avail_list) - np.array(block.require_list)).tolist()
            ps.plan_list.append(Place(space, block))
            ps.volume += block.volume
            cuboid1, cuboid2, cuboid3 = gen_residual_space(space, block)
            ps.space_stack.push(cuboid1, cuboid2, cuboid3)
        else:
            transfer_space(space, ps.space_stack)
    print(ps.avail_list)
    print(ps.volume)
```
