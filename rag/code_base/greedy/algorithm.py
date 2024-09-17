import copy
import numpy as np
from toolbox.draw import draw_packing_result


FILL_RATE = 0.9  # 复合块的最小填充率
AREA_RATE = 0.9  # 可行放置矩形与相应复合块顶部面积比的最小值
TIMES = 2  # 复合块最大复杂度
best_pack_state = None  # 临时的最优放置方案


# 箱子类
class Box:
    def __init__(self, lx, ly, lz, box_type=0):
        # 箱子是问题中被装载的物体，同样用lx,ly,lz三个域来描述它3条边的长度（长，宽，高）
        self.lx, self.ly, self.lz = lx, ly, lz
        # 类型
        self.type = box_type

    def __str__(self):
        return "lx: {}, ly: {}, lz: {}, type: {}".format(self.lx, self.ly, self.lz, self.type)


# 剩余空间类
class Space:
    def __init__(self, x, y, z, lx, ly, lz, origin=None):
        # 对车箱的长宽高分别建立坐标轴：x、y、z，令三维坐标轴的原点为箱子的左后下角，此点的坐标为（0，0，0）
        self.x, self.y, self.z = x, y, z
        # 箱子的装载是在剩余空间中进行的，剩余空间是车厢中的未填充长方体空间，lx、ly、lz 描述了它在3个维度上的长度（长，宽，高）
        self.lx, self.ly, self.lz = lx, ly, lz
        # 表示从哪个剩余空间切割而来
        self.origin = origin

    def __str__(self):
        return "x:{},y:{},z:{},lx:{},ly:{},lz:{}".format(self.x, self.y, self.z, self.lx, self.ly, self.lz)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.z == other.z and self.lx == other.lx and (
               self.ly == other.ly) and self.lz == other.lz


# 存储剩余空间的栈
class Stack:
    def __init__(self):
        # 在基础启发式算法中，剩余空间被组织成堆栈。
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


# 装箱问题
class Problem:
    def __init__(self, container: Space, box_list: list = None, num_list: list = None):
        """
        我们使用一个三元组 (container, box_list, num_list) 来表示该问题。

        :param container: 初始剩余空间
        :param box_list: 箱子向量，指定可用于装载的箱子
        :param num_list: 整数向量，描述了每一种类型箱子的数目
        :rtype: object
        """
        # 初始化参数
        box_list = [] if box_list is None else box_list
        num_list = [] if num_list is None else num_list
        # 容器
        self.container = container
        # 箱子列表
        self.box_list = box_list
        # 箱子对应的数量
        self.num_list = num_list


# 块
class Block:
    def __init__(self, lx, ly, lz, require_list=np.ndarray([]), children: list = None, direction=None):
        """
        块的定义为一个包含许多箱子的长方体。由于 block 中有空隙，block 的顶部有一部分可能由于失去支撑而不能继续放置其它块，
        我们通过可行放置矩形来描述 block 的顶部可以继续放置其它 block 的矩形区域。
        这里我们仅考虑包括 block 顶部左后上角的可行放置矩形，以 block 结构的域 ax, ay 表示其长宽。

        :param lx: 箱子的长
        :param ly: 箱子的宽
        :param lz: 箱子的高
        :param require_list: 整数向量，描述了 block 对各种类型箱子的需求数
        """
        # 初始化参数
        children = [] if children is None else children
        # 长, 宽, 高
        self.lx, self.ly, self.lz = lx, ly, lz
        # 需要的物品数量
        self.require_list = require_list
        # 列表当中箱子的总体积
        self.volume = 0
        # 子块列表，简单块的子块列表为空
        self.children = children
        # 复合块子块的合并方向
        self.direction = direction
        # 顶部可放置矩形尺寸
        self.ax = 0
        self.ay = 0
        # 复杂度，复合次数
        self.times = 0
        # 适应度，块选择时使用
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


# 放置
class Place:
    def __init__(self, space: Space, block: Block):
        # 空间
        self.space = space
        # 块
        self.block = block

    def __eq__(self, other):
        return self.space == other.space and self.block == other.block


# 装箱状态
class PackingState:
    def __init__(self, plan_list: list = None, space_stack: Stack = Stack(), avail_list: list = None):
        # 初始化参数
        plan_list = [] if plan_list is None else plan_list
        avail_list = [] if avail_list is None else avail_list
        # 已生成的装箱方案列表
        self.plan_list = plan_list
        # 剩余空间堆栈
        self.space_stack = space_stack
        # 剩余可用箱体数量
        self.avail_list = avail_list
        # 已装载物品总体积
        self.volume = 0
        # 最终装载物品的总体积的评估值
        self.volume_complete = 0


# 合并块时通用校验项目
def combine_common_check(combine: Block, container: Space, num_list):
    # 合共块尺寸不得大于容器尺寸
    if combine.lx > container.lx:
        return False
    if combine.ly > container.ly:
        return False
    if combine.lz > container.lz:
        return False
    # 合共块需要的箱子数量不得大于箱子总的数量
    if (np.array(combine.require_list) > np.array(num_list)).any():
        return False
    # 合并块的填充体积不得小于最小填充率
    if combine.volume / (combine.lx * combine.ly * combine.lz) < FILL_RATE:
        return False
    # 合并块的顶部可放置矩形必须足够大
    if (combine.ax * combine.ay) / (combine.lx * combine.ly) < AREA_RATE:
        return False
    # 合并块的复杂度不得超过最大复杂度
    if combine.times > TIMES:
        return False
    return True


# 合并块时通用合并项目
def combine_common(a: Block, b: Block, combine: Block):
    # 合并块的需求箱子数量
    combine.require_list = (np.array(a.require_list) + np.array(b.require_list)).tolist()
    # 合并填充体积
    combine.volume = a.volume + b.volume
    # 构建父子关系
    combine.children = [a, b]
    # 合并后的复杂度
    combine.times = max(a.times, b.times) + 1


# 生成简单块
def gen_simple_block(container, box_list, num_list):
    """
    简单块，即相同类型的箱子进行堆叠组成的块，可向 x、y、z 三个方向进行堆叠。
    该函数用于生成简单块，同时定义生成的块中的各个属性：所需箱子数量、长、宽、高、体积、顶部可放置矩形尺寸、简单块的复杂度。

    :param container: 集装箱空间
    :param box_list: 待放置箱子列表
    :param num_list: 剩余箱子数量
    :return: 按体积排序的简单块列表
    """
    block_table = []
    for box in box_list:
        for nx in np.arange(num_list[box.type]) + 1:
            for ny in np.arange(num_list[box.type] / nx) + 1:
                for nz in np.arange(num_list[box.type] / nx / ny) + 1:
                    if box.lx * nx <= container.lx and box.ly * ny <= container.ly and box.lz * nz <= container.lz:
                        # 该简单块需要的立体箱子数量
                        requires = np.full_like(num_list, 0)
                        requires[box.type] = nx * ny * nz
                        # 简单块
                        block = Block(box.lx * nx, box.ly * ny, box.lz * nz, requires)
                        # 顶部可放置矩形
                        block.ax = box.lx * nx
                        block.ay = box.ly * ny
                        # 简单块填充体积
                        block.volume = box.lx * nx * box.ly * ny * box.lz * nz
                        # 简单块复杂度
                        block.times = 0
                        block_table.append(block)
    return sorted(block_table, key=lambda x: x.volume, reverse=True)


# 生成复合块
def gen_complex_block(container, box_list, num_list):
    """
    复合块，即不同简单块的堆叠复合。考虑到约束，以及防止维度爆炸，需要对复合块进行如下限制：
        1. 复合块的大小不大于容器的大小；
        2. 复合块中可以有空隙，但填充率至少要达到 MIN_FILL_RATE；
        3. 按 x 轴方向、按 y 轴方向复合的时候，子块要保证顶部可行放置矩形也能进行复合。在按 z 轴方向复合时，子块要保证复合满足约束 2；
        4. 为控制复杂程度，定义复合块的复杂度：简单块的复杂度为 0，其它复合块的复杂度为其子块的复杂度的最大值加 1，块结构的 times 域描述了复合块的复杂程度，限制生成块的最大复杂次数为 MAX_TIMES；
        5. 受复合块中空隙的影响，复合块顶部有支撑的可行放置矩形可能很小，为了进一步装载，限定可行放置矩形与相应的复合块顶部面积的比至少要达到 MIN_AREA_RATE；
        6. 拥有相同 3 边长度、箱子需求和顶部可行放置矩形的复合块被视为等价块，重复生成的等价块将被忽略；
    满足以上约束的情况下，块数目仍然可能会很大，生成算法将在块数目达到 MaxBlocks 时停止生成。

    该函数用于生成复合块，首先生成简单块，再对简单块按照 x、y、z 三个方向进行复合。
    同时定义生成的块中的各个属性：所需箱子数量、长、宽、高、体积、顶部可放置矩形尺寸、合并方向、子块列表、复合块的复杂度。
    最后，需要按照填充体积对复合块进行排序。

    :param container: 集装箱空间
    :param box_list: 可用块的列表
    :param num_list: 剩余块数量列表
    :return: 按填充体积进行排序的复合块列表
    """
    # 先生成简单块
    block_table = gen_simple_block(container, box_list, num_list)
    for times in range(TIMES):
        new_block_table = []
        # 循环所有简单块，两两配对
        for i in np.arange(0, len(block_table)):
            # 第一个简单块
            a = block_table[i]
            for j in np.arange(0, len(block_table)):
                # 简单块不跟自己复合
                if j == i:
                    continue
                # 第二个简单块
                b = block_table[j]
                # 复杂度满足当前复杂度
                if a.times == times or b.times == times:
                    c = Block(0, 0, 0)
                    # 按x轴方向复合
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
                    # 按y轴方向复合
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
                    # 按z轴方向复合
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
        # 加入新生成的复合块
        block_table = block_table + new_block_table
        # 去重，拥有相同三边长度、物品需求和顶部可放置矩形的复合块被视为等价块，重复生成的等价块将被忽略
        block_table = list(set(block_table))
    # 按填充体积对复合块进行排序
    return sorted(block_table, key=lambda x: x.volume, reverse=True)


# 生成可行块列表
def gen_block_list(space: Space, avail, block_table):
    """
    可行块生成算法，用于从 block_table 中获取适合当前剩余空间的可行block列表。
    可行块生成算法用于生成可行块列表，包含以下约束：
        1. block 中需要的箱子需求数量必须小于当前待装箱的箱子数量；
        2. block 的尺寸必须小于放置空间尺寸；
    该算法扫描 block_table，返回所有能放入剩余空间 space 并且 avail 有足够剩余箱子满足需求的 block。
    由于 block_table 是按 block 中箱子总体积降序排列的，返回的 block_list 也是按箱子总体积降序排列的。

    :param space: 全部空间
    :param avail: 剩余可用箱体的数量
    :param block_table: 按 block 体积降序排列的所有可能 block 的列表
    """
    block_list = []
    for block in block_table:
        # 块中需要的箱子需求数量必须小于当前待装箱的箱子数量
        # 块的尺寸必须小于放置空间尺寸
        if (np.array(block.require_list) <= np.array(avail)).all() and \
                block.lx <= space.lx and block.ly <= space.ly and block.lz <= space.lz:
            block_list.append(block)
    return block_list


# 裁切出新的剩余空间（有稳定性约束）
def gen_residual_space(space: Space, block: Block, box_list: list = None):
    """
    该函数用于裁剪出新的剩余空间。
    首先计算出新的剩余空间，随后将新裁剪出的剩余空间按照三个维度的剩余空间大小进行转移。
    具体而言，如果 x 轴的尺寸大于 y 轴的尺寸，就将该空间归属于 x 轴，反之归属于 y 轴。

    :param space: 总可用空间
    :param block: 放置该块后，裁剪剩余空间
    :param box_list: 块列表（冗余变量）
    :return:
    """
    # 初始化参数
    box_list = [] if box_list is None else box_list
    # 三个维度的剩余尺寸
    rmx = space.lx - block.lx
    rmy = space.ly - block.ly
    rmz = space.lz - block.lz
    # 三个新裁切出的剩余空间（按入栈顺序依次返回）
    if rmx >= rmy:
        # 可转移空间归属于x轴切割空间
        drs_x = Space(space.x + block.lx, space.y, space.z, rmx, space.ly, space.lz, space)
        drs_y = Space(space.x, space.y + block.ly, space.z, block.lx, rmy, space.lz, space)
        drs_z = Space(space.x, space.y, space.z + block.lz, block.ax, block.ay, rmz, None)
        return drs_z, drs_y, drs_x
    else:
        # 可转移空间归属于y轴切割空间
        drs_x = Space(space.x + block.lx, space.y, space.z, rmx, block.ly, space.lz, space)
        drs_y = Space(space.x, space.y + block.ly, space.z, space.lx, rmy, space.lz, space)
        drs_z = Space(space.x, space.y, space.z + block.lz, block.ax, block.ay, rmz, None)
        return drs_z, drs_x, drs_y


# 空间转移
def transfer_space(space: Space, space_stack: Stack):
    """
    该函数用于实现剩余空间的转移，如果仅剩一个空间的话就无法转移，直接弹出。否则，先提取带转移的空间，随后提取目标空间，将可转移的空间转移给目标空间。

    :param space: 总可用空间
    :param space_stack: 装载剩余空间的堆栈
    :return: 未发生转移之前的目标空间
    """
    # 仅剩一个空间的话，直接弹出
    if space_stack.size() <= 1:
        space_stack.pop()
        return None
    # 待转移空间的原始空间
    discard = space
    # 目标空间
    space_stack.pop()
    target = space_stack.top()
    # 将可转移的空间转移给目标空间
    if discard.origin is not None and target.origin is not None and discard.origin == target.origin:
        new_target = copy.deepcopy(target)
        # 可转移空间原先归属于y轴切割空间的情况
        if discard.lx == discard.origin.lx:
            new_target.ly = discard.origin.ly
        # 可转移空间原来归属于x轴切割空间的情况
        elif discard.ly == discard.origin.ly:
            new_target.lx = discard.origin.lx
        else:
            return None
        space_stack.pop()
        space_stack.push(new_target)
        # 返回未发生转移之前的目标空间
        return target
    return None


# 还原空间转移
def transfer_space_back(space: Space, space_stack: Stack, revert_space: Space):
    space_stack.pop()
    space_stack.push(revert_space)
    space_stack.push(space)


# 块放置算法
def place_block(pack_state: PackingState, block: Block):
    """
    箱子的摆放表示为一个剩余空间和块的二元组，(space, block) 表示将 block 放置在剩余空间 space 上的动作。
    而放置动作本身，则是通过将 block 的参考点和剩余空间的参考点重合得到的。
    该函数首先提取栈顶剩余空间。随后更新可用的箱子的数目、更新放置状态、更新体积、得到新的剩余空间 gen_residual_space()，最后，返回本次临时生成的放置。

    :param pack_state: 初始放置状态
    :param block: 要放置的块
    :return: 临时生成的放置
    """
    # 栈顶剩余空间
    space = pack_state.space_stack.pop()
    # 更新可用箱体数目
    pack_state.avail_list = (np.array(pack_state.avail_list) - np.array(block.require_list)).tolist()
    # 更新放置计划
    place = Place(space, block)
    pack_state.plan_list.append(place)
    # 更新体积利用率
    pack_state.volume = pack_state.volume + block.volume
    # 压入新的剩余空间
    cuboid1, cuboid2, cuboid3 = gen_residual_space(space, block)
    pack_state.space_stack.push(cuboid1, cuboid2, cuboid3)
    # 返回临时生成的放置
    return place


# 块移除算法
def remove_block(pack_state: PackingState, block: Block, place: Place, space: Space):
    """
    该函数用于移除块，用于还原以下信息：可用箱体数目、排样计划、体积利用率、剩余空间。

    :param pack_state: 当前装箱方案
    :param block: 要移除的块
    :param place: 当前装箱状态
    :param space: 全部可用空间
    """
    # 还原可用箱体数目
    pack_state.avail_list = (np.array(pack_state.avail_list) + np.array(block.require_list)).tolist()
    # 还原排样计划
    pack_state.plan_list.remove(place)
    # 还原体积利用率
    pack_state.volume = pack_state.volume - block.volume
    # 移除在此之前裁切出的新空间
    for _ in range(3):
        pack_state.space_stack.pop()
    # 还原之前的空间
    pack_state.space_stack.push(space)


# 评价某个块
def estimate(pack_state: PackingState, block_table, search_params):
    """
    该函数用于评价某个块，返回评估值。

    :param pack_state: 待评分放置方案
    :param block_table: 块表
    :param search_params: 相关参数（冗余变量）
    """
    return pack_state.volume


# 查找下一个可行块
def find_next_block(pack_state: PackingState, block_list, block_table, search_params):
    # 最优适应度
    best_fitness = 0
    # 初始化最优块为第一个块（填充体积最大的块）
    best_block = block_list[0]
    # 遍历所有可行块
    for block in block_list:
        # 栈顶空间
        space = pack_state.space_stack.top()
        # 放置块
        place = place_block(pack_state, block)
        # 评价值
        fitness = estimate(pack_state, block_table, search_params)
        # 移除刚才添加的块
        remove_block(pack_state, block, place, space)
        # 更新最优解
        if fitness > best_fitness:
            best_fitness = fitness
            best_block = block

    # 采用贪心算法，直接返回填充体积最大的块
    return best_block


# 基本启发式算法
def basic_heuristic(is_complex: bool, search_params, problem: Problem):
    """
    启发式算法流程如下：
    首先生成复合块: gen_complex_block() 和简单块: gen_simple_block()
    随后初始化状态，包括已生成的装箱方案列表、剩余空间、剩余可用的箱子数量以及已经装载箱子的总体积，最终装载箱子的总体积的评估值。
    进行循环直到剩余空间全部占满：
        生成可行块列表：gen_block_list()
        如果生成列表不为空：
            找到下一个近似最优块：find_next_block()
            弹出栈顶剩余空间
            更新可用的箱子的数量
            更新放置计划
            更新已经利用的体积
            压入更新的剩余空间：gen_residual_space()
        否则，生成列表为空：
            转移剩余空间：transfer_space()
    打印结果，包含剩余的箱子数量以及车厢的利用率。

    :param is_complex: 是否生成复合块列表
    :param search_params: 相关参数
    :param problem: 问题描述类
    """
    if is_complex:
        # 生成复合块列表
        # block_table 是预先生成的表。按 block 体积降序排列的所有可能 block 的列表，用于迅速生成指定剩余空间的可行 block_list 列表。
        # 同时，block_table 将 block 生成算法与装载算法分开，使得更换 block 生成算法变得更容易。
        block_table = gen_complex_block(problem.container, problem.box_list, problem.num_list)
    else:
        # 生成简单块列表
        block_table = gen_simple_block(problem.container, problem.box_list, problem.num_list)
    # 初始化排样状态
    pack_state = PackingState(plan_list=[], avail_list=problem.num_list)
    # 开始时，剩余空间堆栈中只有容器本身
    pack_state.space_stack.push(problem.container)
    # 所有剩余空间均转满，则停止
    while pack_state.space_stack.size() > 0:
        # 从栈顶取一个剩余空间。若有可行块，按照装载序列选择一个块放置在该空间，将未填充空间切割成新的剩余空间加入堆栈
        # 相反，若无可行块，抛弃此剩余空间，如此反复，直至堆栈为空。此过程中，space_stack 表示剩余空间堆栈，整数向量 avail_list 记录剩余箱子数目
        space = pack_state.space_stack.top()
        block_list = gen_block_list(space, pack_state.avail_list, block_table)
        # 在每个装载阶段一个剩余空间被装载，装载分为两种情况：有可行 block，无可行 block
        if block_list:
            # 在有可行 block 时，算法按照块选择算法选择可行 block，然后将未填充空间切割成新的剩余空间
            # 查找下一个近似最优块
            block = find_next_block(pack_state, block_list, block_table, search_params)
            # 弹出顶部剩余空间
            pack_state.space_stack.pop()
            # 更新可用物品数量
            pack_state.avail_list = (np.array(pack_state.avail_list) - np.array(block.require_list)).tolist()
            # 更新排样计划
            pack_state.plan_list.append(Place(space, block))
            # 更新已利用体积
            pack_state.volume = pack_state.volume + block.volume
            # 压入新裁切的剩余空间
            cuboid1, cuboid2, cuboid3 = gen_residual_space(space, block)
            pack_state.space_stack.push(cuboid1, cuboid2, cuboid3)
        else:
            # 在无可行 block 时，当前剩余空间被抛弃，若其中的一部分空间可以被并入当前堆栈中的其他空间，则进行空间转移重新利用这些空间
            # 转移剩余空间，以重新利用
            transfer_space(space, pack_state.space_stack)

    # 打印剩余箱体和已使用容器的体积
    print("剩余箱子数量：{}".format(pack_state.avail_list))
    print("利用率：{}".format(pack_state.volume / (problem.container.lx * problem.container.ly * problem.container.lz)))

    # 绘制排样结果图
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


# 主算法入口
def run_solution(box_list: list, container: list):
    container = Space(*container)
    box_list, num_list = gen_box_num_list(box_list)
    problem = Problem(container, box_list, num_list)
    search_params = dict()
    basic_heuristic(True, search_params, problem)


if __name__ == "__main__":
    CONTAINER = [0, 0, 0, 587, 233, 220]
    BOX_LIST = [[108, 76, 30, 40], [110, 43, 25, 30], [92, 81, 55, 30]]
    run_solution(BOX_LIST, CONTAINER)
