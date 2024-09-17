import numpy as np
from fun_search.draw import visualize_packing


def is_position_valid(L, W, H, x, y, z, l, w, h, placed_boxes):
    """Check if the box fits in the carriage, is supported, and does not overlap with other boxes."""
    # check whether the box inside the container
    if x + l > L or y + w > W or z + h > H:
        return False
    # check whether the box overlapped by other boxes
    if any(px < x + l and px + pl > x and py < y + w and py + pw > y and pz < z + h and pz + ph > z
           for (pl, pw, ph, px, py, pz) in placed_boxes):
        return False

    # check whether the box bottom is supported by other boxes
    if z > 0:
        supported = any(
            pz + ph == z and px <= x < px + pl and px <= x + l <= px + pl and py <= y < py + pw and py <= y + w <= py + pw
            for (pl, pw, ph, px, py, pz) in placed_boxes)
        if not supported:
            return False

    return True


def add_box_and_update_corners(corners, L, W, H, l, w, h, x, y, z):
    """Add a box and update the list of corners with new ones."""
    new_corners = set(corners)
    new_corners.update((x + dx, y + dy, z + dz) for dx in [0, l] for dy in [0, w] for dz in [0, h]
                       if x + dx <= L and y + dy <= W and z + dz <= H)
    return new_corners


def pack_boxes(L, W, H, boxes):
    """Pack boxes into the carriage and return their positions."""
    placed_boxes, failed_boxes = [], []
    corners = {(0, 0, 0)}
    for l, w, h in boxes:
        placed = False
        for corner in list(corners):
            x, y, z = corner
            if is_position_valid(L, W, H, x, y, z, l, w, h, placed_boxes):
                placed_boxes.append((l, w, h, x, y, z))
                corners = add_box_and_update_corners(corners, L, W, H, l, w, h, x, y, z)
                placed = True
                break
        if not placed:
            failed_boxes.append((l, w, h, -1, -1, -1))
    return placed_boxes, failed_boxes


def second_pack_boxes(L, W, H, boxes):
    """Try to pack the failed boxes into the second carriage and return their positions."""
    placed_boxes, failed_boxes = [], []
    corners = {(0, 0, 0)}
    for l, w, h in boxes:
        placed = False
        for corner in list(corners):
            x, y, z = corner
            if is_position_valid(L, W, H, x, y, z, l, w, h, placed_boxes):
                placed_boxes.append((l, w, h, x, y, z))
                corners = add_box_and_update_corners(corners, L, W, H, l, w, h, x, y, z)
                placed = True
                break
        if not placed:
            failed_boxes.append((l, w, h, -1, -1, -1))
    return placed_boxes, failed_boxes


def main():
    # Data for the container and boxes
    L, W, H = 12, 2.33, 2.39
    boxes = [(1.09, 0.5, 0.885)] * 66 + [(1.095, 0.495, 1.48)] * 8 + [(0.915, 0.37, 0.615)] * 80
    # Sort the order of packing in descending order, and pack blocks
    boxes.sort(key=lambda x: x[0] * x[1] * x[2], reverse=True)

    # Pack boxes into the first carriage
    placed_boxes_1, failed_boxes_1 = pack_boxes(L, W, H, boxes)
    # Try to pack remaining boxes into the second carriage
    placed_boxes_2, failed_boxes_2 = second_pack_boxes(L, W, H, [b[:-3] for b in failed_boxes_1])

    # Print the packing result for the first carriage
    for i, (l, w, h, x, y, z) in enumerate(placed_boxes_1):
        print(f"Box {i} ({l:.2f}x{w:.2f}x{h:.2f}) placed in Carriage 1 at ({x:.2f},{y:.2f},{z:.2f})")
    for i, (l, w, h, _, _, _) in enumerate(failed_boxes_1):
        print(f"Box {i + len(placed_boxes_1)} ({l:.2f}x{w:.2f}x{h:.2f}) cannot be placed in Carriage 1")

    # Print the packing result for the second carriage
    for i, (l, w, h, x, y, z) in enumerate(placed_boxes_2):
        print(f"Box {i} ({l:.2f}x{w:.2f}x{h:.2f}) placed in Carriage 2 at ({x:.2f},{y:.2f},{z:.2f})")
    for i, (l, w, h, _, _, _) in enumerate(failed_boxes_2):
        print(f"Box {i + len(placed_boxes_2)} ({l:.2f}x{w:.2f}x{h:.2f}) cannot be placed in Carriage 2")

    # Visualize the packing for both carriages
    visualize_packing(L, W, H, placed_boxes_1)
    visualize_packing(L, W, H, placed_boxes_2)


if __name__ == "__main__":
    main()
