import numpy as np
import plotly.graph_objects as go
from itertools import product
import colorsys


# Function to generate a color gradient
def generate_color(base_color, variation):
    r, g, b = base_color
    # Adjust color slightly
    r = min(max(r + variation[0], 0), 1)
    g = min(max(g + variation[1], 0), 1)
    b = min(max(b + variation[2], 0), 1)
    return f'rgb({int(r * 255)}, {int(g * 255)}, {int(b * 255)})'


# Recursive function to build box positions for drawing
def build_box_position(block, init_pos, box_list):
    if len(block.children) <= 0 and block.times == 0:
        box_idx = (np.array(block.require_list) > 0).tolist().index(True)
        if box_idx > -1:
            box = box_list[box_idx]
            nx = block.lx / box.lx
            ny = block.ly / box.ly
            nz = block.lz / box.lz
            x_list = (np.arange(0, nx) * box.lx).tolist()
            y_list = (np.arange(0, ny) * box.ly).tolist()
            z_list = (np.arange(0, nz) * box.lz).tolist()
            dimensions = (np.array([x for x in product(x_list, y_list, z_list)]) + np.array(
                [init_pos[0], init_pos[1], init_pos[2]])).tolist()
            return sorted([d + [box.lx, box.ly, box.lz] for d in dimensions], key=lambda x: (x[0], x[1], x[2]))
        return []
    pos = []
    for child in block.children:
        pos += build_box_position(child, (init_pos[0], init_pos[1], init_pos[2]), box_list)
        if block.direction == "x":
            init_pos = (init_pos[0] + child.lx, init_pos[1], init_pos[2])
        elif block.direction == "y":
            init_pos = (init_pos[0], init_pos[1] + child.ly, init_pos[2])
        elif block.direction == "z":
            init_pos = (init_pos[0], init_pos[1], init_pos[2] + child.lz)
    return pos


# Draw the packing result
def draw_packing_result(problem, pack_state):
    fig = go.Figure()

    color_map = {}
    base_colors = {
        (lx, ly, lz): colorsys.hsv_to_rgb((hash((lx, ly, lz)) % 360) / 360.0, 0.7, 0.9)
        for lx, ly, lz in set((box.lx, box.ly, box.lz) for box in problem.box_list)
    }

    for p in pack_state.plan_list:
        box_pos = build_box_position(p.block, (p.space.x, p.space.y, p.space.z), problem.box_list)

        for i, bp in enumerate(box_pos):
            size_tuple = (bp[3], bp[4], bp[5])
            if size_tuple not in color_map:
                base_color = base_colors[size_tuple]
                variations = [(0.08, 0, 0), (0, 0.08, 0), (0, 0, 0.08), (-0.08, 0, 0), (0, -0.08, 0), (0, 0, -0.08)]
                color_map[size_tuple] = [generate_color(base_color, var) for var in variations]

            color = color_map[size_tuple][i % len(color_map[size_tuple])]

            x, y, z = bp[0], bp[1], bp[2]
            dx, dy, dz = bp[3], bp[4], bp[5]

            # Add a box to the figure
            fig.add_trace(go.Mesh3d(
                x=[x, x + dx, x + dx, x, x, x + dx, x + dx, x],
                y=[y, y, y + dy, y + dy, y, y, y + dy, y + dy],
                z=[z, z, z, z, z + dz, z + dz, z + dz, z + dz],
                color=color,
                alphahull=0,
                lighting=dict(diffuse=0.9),
                flatshading=True
            ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(nticks=10, range=[0, problem.container.lx]),
            yaxis=dict(nticks=10, range=[0, problem.container.ly]),
            zaxis=dict(nticks=10, range=[0, problem.container.lz]),
        ),
    )

    fig.show()
