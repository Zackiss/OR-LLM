import numpy as np
from fun_search.draw import visualize_packing


def is_position_valid(L, W, H, carriage, l0, w0, h0, l1, w1, h1):
    """Check if the box fits in the carriage and is fully supported."""
    if l0 + l1 > L or w0 + w1 > W or h0 + h1 > H:
        return False
    if np.any(carriage[l0:l0 + l1, w0:w0 + w1, h0:h0 + h1]):
        return False
    if h0 > 0 and not np.all(carriage[l0:l0 + l1, w0:w0 + w1, h0 - 1]):
        return False
    return True


def find_position(L, W, H, carriage, l, w, h):
    """Find a valid position for the box."""
    for i in range(L - l + 1):
        for j in range(W - w + 1):
            for k in range(H - h + 1):
                if is_position_valid(L, W, H, carriage, i, j, k, l, w, h):
                    return i, j, k
    return -1, -1, -1


def pack_boxes(L, W, H, boxes, max_weight):
    """Pack boxes into the carriage and return their positions."""
    carriage = np.zeros((L, W, H), dtype=int)
    positions = []
    current_weight = 0

    for l, w, h in boxes:
        box_weight = l * w * h
        if current_weight + box_weight > max_weight:
            positions.append((l, w, h, -1, -1, -1))
            continue

        pos = find_position(L, W, H, carriage, l, w, h)
        if pos != (-1, -1, -1):
            i, j, k = pos
            carriage[i:i + l, j:j + w, k:k + h] = 1
            positions.append((l, w, h, i, j, k))
            current_weight += box_weight
        else:  # Box cannot be placed
            positions.append((l, w, h, -1, -1, -1))

    return positions, current_weight


def main():
    # Carriage and box dimensions (length, width, height)
    L, W, H = 8, 20, 8
    max_weight = 500
    boxes = [(5, 5, 5)] * 5 + [(3, 3, 3)] * 5 + [(7, 5, 2)] * 3 + [(4, 3, 2)] * 3 + [(6, 6, 6)] * 1 + [(2, 2, 2)] * 6
    # Sort boxes by volume in descending order and pack boxes
    boxes.sort(key=lambda x: x[0] * x[1] * x[2], reverse=True)
    positions, total_weight = pack_boxes(L, W, H, boxes, max_weight)
    # Print result
    for i, (l, w, h, x, y, z) in enumerate(positions):
        print(f"Box {i} ({l}x{w}x{h}) cannot be placed") if x == -1 else print(
            f"Box {i} ({l}x{w}x{h}) placed ({x},{y},{z})")
    print(f"Total weight: {total_weight} kg")
    visualize_packing(L, W, H, positions)


if __name__ == "__main__":
    main()
