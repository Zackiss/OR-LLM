import copy
import numpy as np
from toolbox.draw import draw_packing_result

FILL_RATE = 0.9  
AREA_RATE = 0.9  
TIMES = 2  
best_pack_state = None  


class Box:
    def __init__(self, lx, ly, lz, box_type=0):
        self.lx, self.ly, self.lz = lx, ly, lz
        self.type = box_type

    def __str__(self):
        return "lx: {}, ly: {}, lz: {}, type: {}".format(self.lx, self.ly, self.lz, self.type)


class Space:
    def __init__(self, x, y, z, lx, ly, lz, origin=None):
        self.x, self.y, self.z = x, y, z
        self.lx, self.ly, self.lz = lx, ly, lz
        self.origin = origin

    def __str__(self):
        return "x:{},y:{},z:{},lx:{},ly:{},lz:{}".format(self.x, self.y, self.z, self.lx, self.ly, self.lz)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.z == other.z and self.lx == other.lx and (
                self.ly == other.ly) and self.lz == other.lz


class Stack:
    def __init__(self):
        self.data = []

    def empty(self):
        return len(self.data) == 0

    def not_empty(self):
        return len(self.data) > 0

    def pop(self):
        return self.data.pop() if len(self.data) > 0 else None

    def push(self, *items):
        for item in items:
            self.data.append(item)

    def top(self):
        return self.data[len(self.data) - 1] if len(self.data) > 0 else None

    def clear(self):
        self.data.clear()

    def size(self):
        return len(self.data)


class Problem:
    def __init__(self, container: Space, box_list: list = None, num_list: list = None):
        box_list = [] if box_list is None else box_list
        num_list = [] if num_list is None else num_list
        self.container = container
        self.box_list = box_list
        self.num_list = num_list


class Block:
    def __init__(self, lx, ly, lz, require_list=np.ndarray([]), children: list = None, direction=None):
        children = [] if children is None else children
        self.lx, self.ly, self.lz = lx, ly, lz
        self.require_list = require_list
        self.volume = 0
        self.children = children
        self.direction = direction
        self.ax = 0
        self.ay = 0
        self.times = 0
        self.fitness = 0

    def __str__(self):
        return "lx: %s, ly: %s, lz: %s, volume: %s, ax: %s, ay: %s, times:%s, fitness: %s, require: %s, children: " \
               "%s, direction: %s" % (self.lx, self.ly, self.lz, self.volume, self.ax, self.ay,
                                      self.times, self.fitness, self.require_list, self.children, self.direction)

    def __eq__(self, other):
        return self.lx == other.lx and self.ly == other.ly and self.lz == other.lz and self.ax == other.ax and (
                self.ay == self.ay) and (np.array(self.require_list) == np.array(other.require_list)).all()

    def __hash__(self):
        return hash(",".join([
            str(self.lx), str(self.ly), str(self.lz), str(self.ax), str(self.ay),
            ",".join([str(r) for r in self.require_list])
        ]))


class Place:
    def __init__(self, space: Space, block: Block):
        self.space = space
        self.block = block

    def __eq__(self, other):
        return self.space == other.space and self.block == other.block


class PackingState:
    def __init__(self, plan_list: list = None, space_stack: Stack = Stack(), avail_list: list = None):
        plan_list = [] if plan_list is None else plan_list
        avail_list = [] if avail_list is None else avail_list
        self.plan_list = plan_list
        self.space_stack = space_stack
        self.avail_list = avail_list
        self.volume = 0
        self.volume_complete = 0


def combine_common_check(combine: Block, container: Space, num_list):
    if combine.lx > container.lx:
        return False
    if combine.ly > container.ly:
        return False
    if combine.lz > container.lz:
        return False
    if (np.array(combine.require_list) > np.array(num_list)).any():
        return False
    if combine.volume / (combine.lx * combine.ly * combine.lz) < FILL_RATE:
        return False
    if (combine.ax * combine.ay) / (combine.lx * combine.ly) < AREA_RATE:
        return False
    if combine.times > TIMES:
        return False
    return True


def combine_common(a: Block, b: Block, combine: Block):
    combine.require_list = (np.array(a.require_list) + np.array(b.require_list)).tolist()
    combine.volume = a.volume + b.volume
    combine.children = [a, b]
    combine.times = max(a.times, b.times) + 1


def gen_simple_block(container, box_list, num_list):
    block_table = []
    for box in box_list:
        for nx in np.arange(num_list[box.type]) + 1:
            for ny in np.arange(num_list[box.type] / nx) + 1:
                for nz in np.arange(num_list[box.type] / nx / ny) + 1:
                    if box.lx * nx <= container.lx and box.ly * ny <= container.ly and box.lz * nz <= container.lz:
                        requires = np.full_like(num_list, 0)
                        requires[box.type] = nx * ny * nz
                        block = Block(box.lx * nx, box.ly * ny, box.lz * nz, requires)
                        block.ax = box.lx * nx
                        block.ay = box.ly * ny
                        block.volume = box.lx * nx * box.ly * ny * box.lz * nz
                        block.times = 0
                        block_table.append(block)
    return sorted(block_table, key=lambda x: x.volume, reverse=True)


def gen_complex_block(container, box_list, num_list):
    block_table = gen_simple_block(container, box_list, num_list)
    for times in range(TIMES):
        new_block_table = []
        for i in np.arange(0, len(block_table)):
            a = block_table[i]
            for j in np.arange(0, len(block_table)):
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
                            continue
                    if a.ay == a.ly and b.ay == b.ly and a.lz == b.lz:
                        c.direction = "y"
                        c.ax = min(a.ax, b.ax)
                        c.ay = a.ay + b.ay
                        c.lx = max(a.lx, b.lx)
                        c.ly = a.ly + b.ly
                        c.lz = a.lz
                        combine_common(a, b, c)
                        if combine_common_check(c, container, num_list):
                            new_block_table.append(c)
                            continue
                    if a.ax >= b.lx and a.ay >= b.ly:
                        c.direction = "z"
                        c.ax = b.ax
                        c.ay = b.ay
                        c.lx = a.lx
                        c.ly = a.ly
                        c.lz = a.lz + b.lz
                        combine_common(a, b, c)
                        if combine_common_check(c, container, num_list):
                            new_block_table.append(c)
                            continue
        block_table = block_table + new_block_table
        block_table = list(set(block_table))
    return sorted(block_table, key=lambda x: x.volume, reverse=True)


def gen_block_list(space: Space, avail, block_table):
    block_list = []
    for block in block_table:
        if (np.array(block.require_list) <= np.array(avail)).all() and \
                block.lx <= space.lx and block.ly <= space.ly and block.lz <= space.lz:
            block_list.append(block)
    return block_list


def gen_residual_space(space: Space, block: Block, box_list: list = None):
    box_list = [] if box_list is None else box_list
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


def transfer_space(space: Space, space_stack: Stack):
    if space_stack.size() <= 1:
        space_stack.pop()
        return None
    discard = space
    space_stack.pop()
    target = space_stack.top()
    if discard.origin is not None and target.origin is not None and discard.origin == target.origin:
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


def transfer_space_back(space: Space, space_stack: Stack, revert_space: Space):
    space_stack.pop()
    space_stack.push(revert_space)
    space_stack.push(space)


def place_block(pack_state: PackingState, block: Block):
    space = pack_state.space_stack.pop()
    pack_state.avail_list = (np.array(pack_state.avail_list) - np.array(block.require_list)).tolist()
    place = Place(space, block)
    pack_state.plan_list.append(place)
    pack_state.volume = pack_state.volume + block.volume
    cuboid1, cuboid2, cuboid3 = gen_residual_space(space, block)
    pack_state.space_stack.push(cuboid1, cuboid2, cuboid3)
    return place


def remove_block(pack_state: PackingState, block: Block, place: Place, space: Space):
    pack_state.avail_list = (np.array(pack_state.avail_list) + np.array(block.require_list)).tolist()
    pack_state.plan_list.remove(place)
    pack_state.volume = pack_state.volume - block.volume
    for _ in range(3):
        pack_state.space_stack.pop()
    pack_state.space_stack.push(space)


def estimate(pack_state: PackingState, block_table, search_params):
    return pack_state.volume


def find_next_block(pack_state: PackingState, block_list, block_table, search_params):
    best_fitness = 0
    best_block = block_list[0]
    for block in block_list:
        if not is_valid_pair(block, search_params, pack_state):
            continue
        space = pack_state.space_stack.top()
        place = place_block(pack_state, block)
        fitness = estimate(pack_state, block_table, search_params)
        remove_block(pack_state, block, place, space)
        if fitness > best_fitness:
            best_fitness = fitness
            best_block = block
    return best_block


def is_valid_pair(block, search_params, pack_state):
    pair_type_1_count = block.require_list[search_params['pair_box_type_1']]
    pair_type_2_count = block.require_list[search_params['pair_box_type_2']]
    same_pair_count = pair_type_1_count == pair_type_2_count

    avail_1 = pack_state.avail_list[search_params['pair_box_type_1']]
    avail_2 = pack_state.avail_list[search_params['pair_box_type_2']]
    enough_to_pair = pair_type_1_count <= avail_1 and pair_type_2_count <= avail_2

    return same_pair_count and enough_to_pair


def basic_heuristic(is_complex: bool, search_params, problem: Problem):
    if is_complex:
        block_table = gen_complex_block(problem.container, problem.box_list, problem.num_list)
    else:
        block_table = gen_simple_block(problem.container, problem.box_list, problem.num_list)
    pack_state = PackingState(plan_list=[], avail_list=problem.num_list)
    pack_state.space_stack.push(problem.container)
    while pack_state.space_stack.size() > 0:
        space = pack_state.space_stack.top()
        block_list = gen_block_list(space, pack_state.avail_list, block_table)
        if block_list:
            block = find_next_block(pack_state, block_list, block_table, search_params)
            pack_state.space_stack.pop()
            pack_state.avail_list = (np.array(pack_state.avail_list) - np.array(block.require_list)).tolist()
            pack_state.plan_list.append(Place(space, block))
            pack_state.volume = pack_state.volume + block.volume
            cuboid1, cuboid2, cuboid3 = gen_residual_space(space, block)
            pack_state.space_stack.push(cuboid1, cuboid2, cuboid3)
        else:
            transfer_space(space, pack_state.space_stack)
    print("剩余箱子数量：{}".format(pack_state.avail_list))
    print("利用率：{}".format(pack_state.volume / (problem.container.lx * problem.container.ly * problem.container.lz)))
    draw_packing_result(problem, pack_state)


def gen_box_num_list(data_list: list):
    box_list = []
    num_list = []
    for i in range(0, len(data_list)):
        temp_box_list = Box(data_list[i][0], data_list[i][1], data_list[i][2], i)
        temp_num_list = data_list[i][3]
        box_list.append(temp_box_list)
        num_list.append(temp_num_list)
    return box_list, num_list


def run_solution(box_list: list, container: list):
    PAIR_BOX_TYPE_1 = 0  
    PAIR_BOX_TYPE_2 = 1  

    min_pair_count = min(box_list[PAIR_BOX_TYPE_1][3], box_list[PAIR_BOX_TYPE_2][3])

    box_list[PAIR_BOX_TYPE_1][3] = min_pair_count
    box_list[PAIR_BOX_TYPE_2][3] = min_pair_count

    container = Space(*container)
    box_list, num_list = gen_box_num_list(box_list)

    problem = Problem(container, box_list, num_list)

    search_params = {
        'pair_box_type_1': PAIR_BOX_TYPE_1,
        'pair_box_type_2': PAIR_BOX_TYPE_2,
        'pair_count': min_pair_count
    }
    basic_heuristic(True, search_params, problem)


if __name__ == "__main__":
    CONTAINER = [0, 0, 0, 587, 233, 220]
    BOX_LIST = [[108, 76, 30, 24], [110, 43, 25, 9], [92, 81, 55, 8], [81, 33, 28, 11], [120, 99, 73, 11],
                [111, 70, 48, 10], [98, 72, 46, 12], [95, 66, 31, 9]]
    run_solution(BOX_LIST, CONTAINER)