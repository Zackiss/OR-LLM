### 1. 要求

物流公司在流通过程中，需要将打包完毕的箱子装入到一个货车的车厢中，为了提高物流效率，需要将车厢尽量填满，显然，车厢如果能被 $100\%$ 填满是最优的，但通常认为，车厢能够填满 $85\%$，认为装箱是优化的。

设车厢为长方形，其长宽高分别为 $L,W,H$；共有 $n$ 个箱子，箱子也为长方形，第 $i$ 个箱子的长宽高为 $l_i$, $w_i$, $h_i$（ $n$ 个箱子的体积总和是要远远大于车厢的体积），做以下假设和要求：

1. 长方形的车厢共有 8 个角，并设靠近驾驶室并位于下端的一个角的坐标为 $(0,0,0)$ ，车厢共6个面，其中长的4个面，以及靠近驾驶室的面是封闭的，只有一个面是开着的，用于工人搬运箱子

2. 需要计算出每个箱子在车厢中的坐标，即每个箱子摆放后，其和车厢坐标为 $(0,0,0)$ 的角相对应的角在车厢中的坐标，并计算车厢的填充率。

### 2. 带有方向限制判断的贪心算法求解
带有方向限制判断的贪心算法，流程如下：
1. 箱子按照先上后前再右的顺序摆放。

2. 取出当前可行块。在块选择算法过程中，可以采用贪心算法，直接返回填充体积最大的块，由于可行块列表已经按照体积降序排列，实际上算法选择的块总是列表的第一个元素。

3. 第一个到达的箱子摆在车厢的左后角，第二到达的箱子检测是否可以摆在已有箱子的上面，若不能则摆在前一个箱子的前面。

4. 直到下一个箱子无法摆放到已有箱子的上面，也无法再往前摆，则开始摆第二列。第一列的空间就此舍弃不再摆放。可以省略由于摆放顺序引起的复杂多样的摆放限制因素，提高效率。第一列中最宽的箱子会形成第一列的边界，当第二列箱子进行摆放时不允许超过此边界。


### 3. 代码示例
```python
import copy
import numpy as np
import random
import time

# Define constants
MIN_FILL_RATE = 0.8
MIN_AREA_RATE = 0.8
MAX_TIMES = 2
MAX_DEPTH = 3
MAX_BRANCH = 2

# Temporary best placement solution
tmp_best_ps = None

# Stack class for storing remaining spaces
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
        return self.data[-1] if len(self.data) > 0 else None

    def clear(self):
        self.data.clear()

    def size(self):
        return len(self.data)

# Box class
class Box:
    def __init__(self, lx, ly, lz, type=0):
        self.lx = lx
        self.ly = ly
        self.lz = lz
        self.type = type

    def __str__(self):
        return f"lx: {self.lx}, ly: {self.ly}, lz: {self.lz}, type: {self.type}"

# Space class
class Space:
    def __init__(self, x, y, z, lx, ly, lz, origin=None):
        self.x = x
        self.y = y
        self.z = z
        self.lx = lx
        self.ly = ly
        self.lz = lz
        self.origin = origin

    def __str__(self):
        return f"x:{self.x},y:{self.y},z:{self.z},lx:{self.lx},ly:{self.ly},lz:{self.lz}"

    def __eq__(self, other):
        return (self.x == other.x and self.y == other.y and self.z == other.z and
                self.lx == other.lx and self.ly == other.ly and self.lz == other.lz)

# Problem class
class Problem:
    def __init__(self, container: Space, box_list=[], num_list=[]):
        self.container = container
        self.box_list = box_list
        self.num_list = num_list

# Block class
class Block:
    def __init__(self, lx, ly, lz, require_list=[], children=[], direction=None):
        self.lx = lx
        self.ly = ly
        self.lz = lz
        self.require_list = require_list
        self.volume = 0
        self.children = children
        self.direction = direction
        self.ax = 0
        self.ay = 0
        self.times = 0
        self.fitness = 0

    def __str__(self):
        return (f"lx: {self.lx}, ly: {self.ly}, lz: {self.lz}, volume: {self.volume}, ax: {self.ax}, ay: {self.ay}, "
                f"times: {self.times}, fitness: {self.fitness}, require: {self.require_list}, children: {self.children}, direction: {self.direction}")

    def __eq__(self, other):
        return (self.lx == other.lx and self.ly == other.ly and self.lz == other.lz and
                self.ax == other.ax and self.ay == other.ay and
                (np.array(self.require_list) == np.array(other.require_list)).all())

    def __hash__(self):
        return hash(",".join([str(self.lx), str(self.ly), str(self.lz), str(self.ax), str(self.ay), ",".join([str(r) for r in self.require_list])]))

# Place class
class Place:
    def __init__(self, space: Space, block: Block):
        self.space = space
        self.block = block

    def __eq__(self, other):
        return self.space == other.space and self.block == other.block

# PackingState class
class PackingState:
    def __init__(self, plan_list=[], space_stack: Stack = Stack(), avail_list=[]):
        self.plan_list = plan_list
        self.space_stack = space_stack
        self.avail_list = avail_list
        self.volume = 0
        self.volume_complete = 0

# Common check for combining blocks
def combine_common_check(combine: Block, container: Space, num_list):
    if combine.lx > container.lx or combine.ly > container.ly or combine.lz > container.lz:
        return False
    if (np.array(combine.require_list) > np.array(num_list)).any():
        return False
    if combine.volume / (combine.lx * combine.ly * combine.lz) < MIN_FILL_RATE:
        return False
    if (combine.ax * combine.ay) / (combine.lx * combine.ly) < MIN_AREA_RATE:
        return False
    if combine.times > MAX_TIMES:
        return False
    return True

# Common combine logic
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
                for nz in range(1, num_list[box.type] // nx // ny + 1):
                    if box.lx * nx <= container.lx and box.ly * ny <= container.ly and box.lz * nz <= container.lz:
                        requires = np.zeros_like(num_list)
                        requires[box.type] = nx * ny * nz
                        block1 = Block(box.lx * nx, box.ly * ny, box.lz * nz, requires)
                        block2 = Block(box.ly * nx, box.lx * ny, box.lz * nz, requires)
                        block1.ax = box.lx * nx
                        block1.ay = box.ly * ny
                        block2.ax = box.ly * nx
                        block2.ay = box.lx * ny
                        block1.volume = box.lx * nx * box.ly * ny * box.lz * nz
                        block2.volume = box.lx * nx * box.ly * ny * box.lz * nz
                        block1.times = 0
                        block2.times = 0
                        block_table.append(block1)
                        block_table.append(block2)
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

# Generate feasible block list
def gen_block_list(space: Space, avail, block_table):
    block_list = []
    for block in block_table:
        if (np.array(block.require_list) <= np.array(avail)).all() and \
                block.lx <= space.lx and block.ly <= space.ly and block.lz <= space.lz:
            block_list.append(block)
    return block_list

# Cut new residual space
def gen_residual_space(space: Space, block: Block):
    rmx = space.lx - block.lx
    rmy = space.ly - block.ly
    rmz = space.lz - block.lz
    drs_x = Space(space.x + block.lx, space.y, space.z, rmx, space.ly, space.lz, space)
    drs_y = Space(space.x, space.y + block.ly, space.z, block.lx, rmy, space.lz, space)
    drs_z = Space(space.x, space.y, space.z + block.lz, block.ax, block.ay, rmz, None)
    return drs_z, drs_y, drs_x if rmx >= rmy else (drs_z, drs_x, drs_y)

# Transfer space
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

# Reverse transfer space
def transfer_space_back(space: Space, space_stack: Stack, revert_space: Space):
    space_stack.pop()
    space_stack.push(revert_space)
    space_stack.push(space)

# Place block algorithm
def place_block(ps: PackingState, block: Block):
    space = ps.space_stack.pop()
    ps.avail_list = (np.array(ps.avail_list) - np.array(block.require_list)).tolist()
    place = Place(space, block)
    ps.plan_list.append(place)
    ps.volume += block.volume
    cuboid1, cuboid2, cuboid3 = gen_residual_space(space, block)
    ps.space_stack.push(cuboid1, cuboid2, cuboid3)
    return place

# Remove block algorithm
def remove_block(ps: PackingState, block: Block, place: Place, space: Space):
    ps.avail_list = (np.array(ps.avail_list) + np.array(block.require_list)).tolist()
    ps.plan_list.remove(place)
    ps.volume -= block.volume
    for _ in range(3):
        ps.space_stack.pop()
    ps.space_stack.push(space)

# Complete placement solution
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

# Depth-first search with depth limit
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

# Evaluate a block
def estimate(ps: PackingState, block_table, search_params):
    global tmp_best_ps
    tmp_best_ps = PackingState([], Stack(), [])
    depth_first_search(ps, MAX_DEPTH, MAX_BRANCH, block_table)
    return tmp_best_ps.volume_complete

# Find next feasible block
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

# Generate a block for the first box
def gen_one_block(cur_box, num_list):
    box = Box(cur_box.lx, cur_box.ly, cur_box.lz, cur_box.type)
    xyz = sorted([int(box.lx), int(box.ly), int(box.lz)], reverse=True)
    box.lx, box.ly, box.lz = xyz
    requires = np.zeros_like(num_list)
    requires[box.type] = 1
    block = Block(box.lx, box.ly, box.lz, requires)
    block.ax = box.lx
    block.ay = box.ly
    block.volume = box.lx * box.ly * box.lz
    block.times = 0
    return block

# Generate six blocks for other boxes
def gen_six_block(curr_box, num_list):
    box = Box(curr_box.lx, curr_box.ly, curr_box.lz, curr_box.type)
    cur_block_table = []
    box_xyz = [int(box.lx), int(box.ly), int(box.lz)]
    xyz_type = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]
    for i in range(6):
        cur_box = Box(curr_box.lx, curr_box.ly, curr_box.lz, curr_box.type)
        cur_box.lx = box_xyz[xyz_type[i][0]]
        cur_box.ly = box_xyz[xyz_type[i][1]]
        cur_box.lz = box_xyz[xyz_type[i][2]]
        requires = np.zeros_like(num_list)
        requires[cur_box.type] = 1
        block = Block(cur_box.lx, cur_box.ly, cur_box.lz, requires)
        block.ax = cur_box.lx
        block.ay = cur_box.ly
        block.volume = cur_box.lx * cur_box.ly * cur_box.lz
        block.times = 0
        cur_block_table.append(block)
    return cur_block_table

# Generate new spaces with stability constraint
def gen_xz_space(space: Space, block: Block):
    rmx = space.lx - block.lx
    rmy = space.ly - block.ly
    rmz = space.lz - block.lz
    drs_x = Space(space.x + block.lx, space.y, space.z, rmx, space.ly, space.lz, space)
    drs_y = Space(space.x, space.y + block.ly, space.z, block.lx, rmy, space.lz, space)
    drs_z = Space(space.x, space.y, space.z + block.lz, block.ax, block.ay, rmz, None)
    return drs_x, drs_y, drs_z

# Basic heuristic algorithm
def basic_heuristic(is_complex, search_params, problem: Problem, type_list):    ps = PackingState(avail_list=problem.num_list)
    ps.space_stack.push(problem.container)
    newcontainer = Space(problem.container.x, problem.container.y, problem.container.z, problem.container.lx, problem.container.ly, problem.container.lz, None)
    block_table = []
    for i in range(len(type_list)):
        st = time.time()
        cur_box = problem.box_list[type_list[i]]
        if i == 0:
            block_table.append(gen_one_block(cur_box, problem.num_list))
            space = ps.space_stack.top()
            block_list = gen_block_list(space, ps.avail_list, block_table)
            block_table = []
            if block_list:
                block = block_list[0]
                ps.space_stack.pop()
                ps.avail_list = (np.array(ps.avail_list) - np.array(block.require_list)).tolist()
                ps.plan_list.append(Place(space, block))
                ps.volume += block.volume
                print(f"Coordinate: （{space.x},{space.y},{space.z}）, Edge point: {space.x + block.lx},{space.y + block.ly},{space.z + block.lz})")
                cuboidx, cuboidy, cuboidz = gen_xz_space(space, block)
                if cuboidx.lx * cuboidx.ly < cuboidz.lx * cuboidz.ly:
                    ps.space_stack.push(cuboidz, cuboidx)
                else:
                    ps.space_stack.push(cuboidx, cuboidz)
            else:
                print("No more position avaliable")
        else:
            block_table = gen_six_block(cur_box, problem.num_list)
            if is_complex:
                block_table += gen_complex_block(newcontainer, [cur_box], problem.num_list)
            space = ps.space_stack.top()
            block_list = gen_block_list(space, ps.avail_list, block_table)
            if block_list:
                block = find_next_block(ps, block_list, block_table, search_params)
                place_block(ps, block)
                print(f"Coordinate: （{space.x},{space.y},{space.z}）, Edge point: {space.x + block.lx},{space.y + block.ly},{space.z + block.lz})")
            else:
                print("No more position avaliable")

        et = time.time()
        print(f"Time used {et - st:.2f}sec")

    print(f"\nCapacity rate: {ps.volume / (problem.container.lx * problem.container.ly * problem.container.lz):.2%}")

# Example usage
container = Space(0, 0, 0, 100, 100, 100)
box_list = [Box(10, 10, 10, 0), Box(20, 20, 20, 1)]
num_list = [10, 5]
problem = Problem(container, box_list, num_list)
type_list = [0, 1]
basic_heuristic(True, {}, problem, type_list)
```
