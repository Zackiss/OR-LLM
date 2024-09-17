import random
import plotly.graph_objects as go
import plotly.io as pio


color_map = {}


def random_color():
    """Generate a random base color in hexadecimal format."""
    r, g, b = [random.randint(0, 255) for _ in range(3)]
    return r, g, b


def vibrance_adjust(color, amount=20):
    """Adjust the color to add vibrance with a small random amount."""
    r, g, b = color
    r = min(max(r + random.randint(-amount, amount), 0), 255)
    g = min(max(g + random.randint(-amount, amount), 0), 255)
    b = min(max(b + random.randint(-amount, amount), 0), 255)
    return r, g, b


def register_color(l, w, h):
    """Register or retrieve a color for a given box size, with added vibrance."""
    if (l, w, h) not in color_map:
        base_color = random_color()
        color_map[(l, w, h)] = base_color
    vibrant_color = vibrance_adjust(color_map[(l, w, h)])
    return "#{:02x}{:02x}{:02x}".format(*vibrant_color)


def visualize_packing(L, W, H, positions):
    fig = go.Figure()

    for idx, (l, w, h, x, y, z) in enumerate(positions):
        if x != -1:
            fig.add_trace(go.Mesh3d(
                x=[x, x + l, x + l, x, x, x + l, x + l, x],
                y=[y, y, y + w, y + w, y, y, y + w, y + w],
                z=[z, z, z, z, z + h, z + h, z + h, z + h],
                alphahull=0,
                lighting=dict(diffuse=0.9),
                flatshading=True,
                color=register_color(l, w, h),
                name=f'Box {idx + 1}: {l:.2f}x{w:.2f}x{h:.2f}'
            ))

    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            xaxis=dict(range=[0, L], autorange=False),
            yaxis=dict(range=[0, W], autorange=False),
            zaxis=dict(range=[0, H], autorange=False),
            aspectmode='data'
        ),
    )

    return fig
