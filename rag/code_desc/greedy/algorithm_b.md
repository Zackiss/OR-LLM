### 1. 要求

物流公司在流通过程中，需要将打包完毕的箱子装入到一个货车的车厢中，为了提高物流效率，需要将车厢尽量填满，显然，车厢如果能被 $100\%$ 填满是最优的，但通常认为，车厢能够填满 $85\%$，认为装箱是优化的。

设车厢为长方形，其长宽高分别为 $L,W,H$；共有 $n$ 个箱子，箱子也为长方形，第 $i$ 个箱子的长宽高为 $l_i$, $w_i$, $h_i$（ $n$ 个箱子的体积总和是要远远大于车厢的体积），做以下假设和要求：

1. 长方形的车厢共有 8 个角，并设靠近驾驶室并位于下端的一个角的坐标为 $(0,0,0)$ ，车厢共6个面，其中长的4个面，以及靠近驾驶室的面是封闭的，只有一个面是开着的，用于工人搬运箱子

2. 需要计算出每个箱子在车厢中的坐标，即每个箱子摆放后，其和车厢坐标为 $(0,0,0)$ 的角相对应的角在车厢中的坐标，并计算车厢的填充率。

### 2. 有复合块约束的贪心块装载求解

采用基于块装载的启发式方法。块是装载中使用的启发式算法中，装载的最小单位，是一个包含一个或多个箱子的长方体，装箱后，一个可行块将会被切割，剩余空间将会被组织为堆栈，并考虑是否可以并入到堆栈中的其他空间，从而形成新的可行块。

使用块生成算法进行生成，有简单块与复合块两个概念。简单块是由同一朝向的同种类型的箱子堆叠而成的，箱子和箱子之间没有空隙，堆叠结果为一个长方体。复合块是通过不断复合简单块得到的，简单块是最基本的复合块。

两个复合块 $a$ 和 $b$，可以按如下 $3$ 种方式进行复合，得到复合块 $c$ ：按 $x$ 轴方向复合，按 $y$ 轴方向复合，按 $z$ 轴方向复合。$c$ 的大小是包含 $a$ 和 $b$ 的最小长方体。

因此，复合块的数量将是箱子数目的指数级。而且，生成块中可能有很多未利用空间，非常不利于装载. 所以对复合块施加一定的限制是必要的，生成的复合块必须满足下面条件：

1. 复合块的大小不大于容器的大小。

2. 复合块中可以有空隙，但它的填充率至少要达到 `MIN_FILL_RATE`。

3. 受复合块中空隙的影响，复合块顶部有支撑的可行放置矩形可能小，为了进一步装载，限定可行放置矩形与相应的复合块顶部面积之比至少要达到  `MIN_AREA_RATE`。

4. 为控制复合块的复杂程度，定义复合块的复杂度：简单块的复杂度为 $0$，其它复合块的复杂度为其子块的复杂度的最大值加 $1$。块结构的 `times` 域描述了复合块的复杂程度，限制生成块的最大复杂次数为 `MAX_TIMES`。

5. 按 $x$ 轴方向、按 $y$ 轴方向复合的时候，子块要保证顶部可行放置矩形也能进行复合。在按 $z$ 轴方向复合时，子块要保证复合满足稳定性约束，即禁止箱子悬空。具体的复合要求和结果如下表所示。

6. 拥有相同三边长度、箱子需求和顶部可行放置矩形的复合块被视为等价块，重复生成的等价块将被忽略。

7. 在满足以上约束的情况下，块数目仍然可能会很大，生成算法将在块数目达到 `MAX_BLOCKS` 时停止生成。

在每个装载阶段一个剩余空间被装载，装载分为有无可行块两种情况：
1. 有可行块时，算法按照块选择算法选择可行块，然后将未填充空间切割成新的剩余空间。
2. 无可行块时，当前剩余空间被抛弃.若其中一部分空间可被并入当前堆栈中的其他空间，则进行空间转移重新利用这些空间。

在块选择时，使用贪心算法，选择当前最大的可行块进行装填。

### 3. 代码示例
```python
import sys
import time
import math
import numpy as np

MAX_GAP = 0

# Plane class
class Plane:
    def __init__(self, x, y, z, lx, ly, height_limit=0):
        self.x, self.y, self.z = x, y, z
        self.lx, self.ly = lx, ly
        self.height_limit = height_limit

    def __eq__(self, other):
        return (self.x, self.y, self.z, self.lx, self.ly) == (other.x, other.y, other.z, other.lx, other.ly)

    def adjacent_with(self, other):
        if self.z != other.z:
            return False, None, None

        my_center = (self.x + self.lx / 2, self.y + self.ly / 2)
        other_center = (other.x + other.lx / 2, other.y + other.ly / 2)

        x_adjacent_measure = self.lx / 2 + other.lx / 2
        y_adjacent_measure = self.ly / 2 + other.ly / 2

        if x_adjacent_measure + MAX_GAP >= math.fabs(my_center[0] - other_center[0]) >= x_adjacent_measure:
            if self.y == other.y and self.ly == other.ly:
                ms1 = Plane(min(self.x, other.x), self.y, self.z, self.lx + other.lx, self.ly)
                return True, ms1, None
            if self.y == other.y:
                ms1 = Plane(min(self.x, other.x), self.y, self.z, self.lx + other.lx, min(self.ly, other.ly))
                ms2 = Plane(self.x, self.y + other.ly, self.z, self.lx, self.ly - other.ly) if self.ly > other.ly else Plane(other.x, self.y + self.ly, self.z, other.lx, other.ly - self.ly)
                return True, ms1, ms2
            if self.y + self.ly == other.y + other.ly:
                ms1 = Plane(min(self.x, other.x), max(self.y, other.y), self.z, self.lx + other.lx, min(self.ly, other.ly))
                ms2 = Plane(self.x, self.y, self.z, self.lx, self.ly - other.ly) if self.ly > other.ly else Plane(other.x, other.y, self.z, other.lx, other.ly - self.ly)
                return True, ms1, ms2

        if y_adjacent_measure + MAX_GAP >= math.fabs(my_center[1] - other_center[1]) >= y_adjacent_measure:
            if self.x == other.x and self.lx == other.lx:
                ms1 = Plane(self.x, min(self.y, other.y), self.z, self.lx, self.ly + other.ly)
                return True, ms1, None
            if self.x == other.x:
                ms1 = Plane(self.x, min(self.y, other.y), self.z, min(self.lx, other.lx), self.ly + other.ly)
                ms2 = Plane(self.x + other.lx, self.y, self.z, self.lx - other.lx, self.ly) if self.lx > other.lx else Plane(self.x + self.lx, other.y, self.z, other.lx - self.lx, other.ly)
                return True, ms1, ms2
            if self.x + self.lx == other.x + other.lx:
                ms1 = Plane(max(self.x, other.x), min(self.y, other.y), self.z, min(self.lx, other.lx), self.ly + other.ly)
                ms2 = Plane(self.x, self.y, self.z, self.lx - other.lx, self.ly) if self.lx > other.lx else Plane(other.x, other.y, self.z, other.lx - self.lx, other.ly)
                return True, ms1, ms2
        return False, None, None

# Problem class
class Problem:
    def __init__(self, container: Plane, height_limit=sys.maxsize, box_list=[], num_list=[], rotate=False):
        self.container = container
        self.height_limit = height_limit
        self.box_list = box_list
        self.num_list = num_list
        self.rotate = rotate

# PackingState class
class PackingState:
    def __init__(self, plane_list=[], avail_list=[]):
        self.plan_list = []
        self.avail_list = avail_list
        self.plane_list = plane_list
        self.spare_plane_list = []
        self.volume = 0

# Select plane with lowest z-coordinate
def select_plane(ps: PackingState):
    min_z = min([p.z for p in ps.plane_list])
    temp_planes = [p for p in ps.plane_list if p.z == min_z]
    if len(temp_planes) == 1:
        return temp_planes[0]

    min_area = min([p.lx * p.ly for p in temp_planes])
    temp_planes = [p for p in temp_planes if p.lx * p.ly == min_area]
    if len(temp_planes) == 1:
        return temp_planes[0]

    min_narrow = min([p.lx/p.ly if p.lx <= p.ly else p.ly/p.lx for p in temp_planes])
    new_temp_planes = [p for p in temp_planes if (p.lx/p.ly if p.lx <= p.ly else p.ly/p.lx) == min_narrow]
    if len(new_temp_planes) == 1:
        return new_temp_planes[0]

    min_x = min([p.x for p in new_temp_planes])
    new_temp_planes = [p for p in new_temp_planes if p.x == min_x]
    if len(new_temp_planes) == 1:
        return new_temp_planes[0]

    min_y = min([p.y for p in new_temp_planes])
    new_temp_planes = [p for p in new_temp_planes if p.y == min_y]
    return new_temp_planes[0]

# Disable plane
def disable_plane(ps: PackingState, plane: Plane):
    ps.plane_list.remove(plane)
    ps.spare_plane_list.append(plane)

# Generate block list
def gen_block_list(plane: Plane, avail, block_table, max_height):
    block_list = []
    for block in block_table:
        if (np.array(block.require_list) <= np.array(avail)).all() and block.lx <= plane.lx and block.ly <= plane.ly and block.lz <= max_height - plane.z:
            block_list.append(block)
    return block_list

# Find next block
def find_block(plane: Plane, block_list, ps: PackingState):
    plane_area = plane.lx * plane.ly
    min_residual_area = min([plane_area - b.lx * b.ly for b in block_list])
    candidate = [b for b in block_list if plane_area - b.lx * b.ly == min_residual_area]
    return sorted(candidate, key=lambda x: x.volume, reverse=True)[0]

# Fill plane with block
def fill_block(ps: PackingState, plane: Plane, block):
    ps.avail_list = (np.array(ps.avail_list) - np.array(block.require_list)).tolist()
    ps.plan_list.append(Place(plane, block))
    ps.volume += block.volume
    rs_top, rs1, rs2 = gen_new_plane(plane, block)
    ps.plane_list.remove(plane)
    if rs_top:
        ps.plane_list.append(rs_top)
    if rs1:
        ps.plane_list.append(rs1)
    if rs2:
        ps.plane_list.append(rs2)

# Generate new plane
def gen_new_plane(plane: Plane, block):
    rs_top = Plane(plane.x, plane.y, plane.z + block.lz, block.lx, block.ly)
    if block.lx == plane.lx and block.ly == plane.ly:
        return rs_top, None, None
    if block.lx == plane.lx:
        return rs_top, Plane(plane.x, plane.y + block.ly, plane.z, plane.lx, plane.ly - block.ly), None
    if block.ly == plane.ly:
        return rs_top, Plane(plane.x + block.lx, plane.y, plane.z, plane.lx - block.lx, block.ly), None

    rsa1 = Plane(plane.x, plane.y + block.ly, plane.z, plane.lx, plane.ly - block.ly)
    rsa2 = Plane(plane.x + block.lx, plane.y, plane.z, plane.lx - block.lx, block.ly)
    rsb1 = Plane(plane.x, plane.y + block.ly, plane.z, block.lx, plane.ly - block.ly)
    rsb2 = Plane(plane.x + block.lx, plane.y, plane.z, plane.lx - block.lx, plane.ly)

    rsa_bigger = rsa1 if rsa1.lx * rsa1.ly >= rsa2.lx * rsa2.ly else rsa2
    rsb_bigger = rsb1 if rsb1.lx * rsb1.ly >= rsb2.lx * rsb2.ly else rsb2

    if rsa_bigger.lx * rsa_bigger.ly >= rsb_bigger.lx * rsb_bigger.ly:
        return rs_top, rsa1, rsa2
    else:
        return rs_top, rsb1, rsb2

# Basic heuristic algorithm
def basic_heuristic(problem: Problem):
    ps_final = None
    used_radio_max = 0

    for i in range(6):
        block_table = gen_simple_block(problem.container, problem.box_list, problem.num_list, problem.height_limit, problem.rotate, i)
        ps = PackingState(avail_list=problem.num_list)
        ps.plane_list.append(Plane(problem.container.x, problem.container.y, problem.container.z, problem.container.lx, problem.container.ly))
        max_used_high = 0

        while ps.plane_list:
            plane = select_plane(ps)
            block_list = gen_block_list(plane, ps.avail_list, block_table, problem.height_limit)
            if block_list:
                block = find_block(plane, block_list, ps)
                fill_block(ps, plane, block)
                max_used_high = max(max_used_high, plane.z + block.lz)
            else:
                merge_plane(ps, plane, block_table, problem.height_limit)

        used_volume = problem.container.lx * problem.container.ly * max_used_high
        used_ratio = round(float(ps.volume) * 100 / float(used_volume), 3) if used_volume > 0 else 0
        if used_ratio > used_radio_max:
            used_radio_max = used_ratio
            ps_final = ps

    return ps_final.avail_list, used_radio_max, ps_final

# Import data is assumed to be implemented
def import_data():
    return container_list, box_list, num_list

# Main function
if __name__ == "__main__":
    container_list, box_list, num_list = import_data()
    for i in range(len(container_list)):
        container = container_list[i][0]
        box_list_instance = box_list[i]
        num_list_instance = num_list[i]
        problem = Problem(container=container, height_limit=container.height_limit, box_list=box_list_instance, num_list=num_list_instance, rotate=True)
        new_avail_list, used_ratio, _ = basic_heuristic(problem)
        print('Case', i+1, 'Used Ratio:', used_ratio, '%')
```
