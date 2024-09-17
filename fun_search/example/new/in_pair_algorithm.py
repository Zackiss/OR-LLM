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
        supported = any(pz + ph == z and px <= x < px + pl and px <= x + l <= px + pl and py <= y < py + pw and py <= y + w <= py + pw
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
    
    box_type_counts = {'small': 0, 'large': 0, 'other': 0}  # Initialize counters
    small_count, large_count = 0, 0  # Initialize small and large type counts
    stop_placement = False  # Control variable for stopping placement of paired boxes

    for box in boxes:
        l, w, h, type = box
        placed = False

        for corner in list(corners):
            x, y, z = corner
            if is_position_valid(L, W, H, x, y, z, l, w, h, placed_boxes):
                placed_boxes.append((l, w, h, x, y, z))
                corners = add_box_and_update_corners(corners, L, W, H, l, w, h, x, y, z)
                placed = True
                
                if type == 'small':
                    small_count += 1
                    box_type_counts['small'] += 1
                elif type == 'large':
                    large_count += 1
                    box_type_counts['large'] += 1
                else:
                    box_type_counts['other'] += 1
                break
                
        if not placed:
            if type == 'small' or type == 'large':
                stop_placement = True
            failed_boxes.append((l, w, h, -1, -1, -1))

        if stop_placement and (type == 'small' or type == 'large'):
            break
    
    return placed_boxes, failed_boxes, box_type_counts


def main():
    # Data for the container and boxes
    L, W, H = 12, 2.33, 2.39
    small_boxes = [(1.09, 0.5, 0.885, 'small')] * 66
    large_boxes = [(1.095, 0.495, 1.48, 'large')] * 15
    other_boxes = [(0.915, 0.37, 0.615, 'other')] * 80
    
    # Interleave small and large boxes for alternating pattern
    interleaved_boxes = [box for pair in zip(small_boxes, large_boxes) for box in pair]
    
    # Add other boxes to the list
    all_boxes = interleaved_boxes[:len(small_boxes) + len(large_boxes)] + other_boxes
    
    # Pack boxes
    placed_boxes, failed_boxes, box_type_counts = pack_boxes(L, W, H, all_boxes)
    
    # Print the packing result
    for i, (l, w, h, x, y, z) in enumerate(placed_boxes):
        print(f"Box {i} ({l:.2f}x{w:.2f}x{h:.2f}) placed at ({x:.2f},{y:.2f},{z:.2f})")
    for i, (l, w, h, _, _, _) in enumerate(failed_boxes):
        print(f"Box {i + len(placed_boxes)} ({l:.2f}x{w:.2f}x{h:.2f}) cannot be placed")
    
    # Print the count of each box type
    print(f"Small boxes placed: {box_type_counts['small']}")
    print(f"Large boxes placed: {box_type_counts['large']}")
    print(f"Other boxes placed: {box_type_counts['other']}")
    
    # Visualize the packing
    visualize_packing(L, W, H, placed_boxes)


if __name__ == "__main__":
    main()