### 1. 要求

物流公司在流通过程中，需要将打包完毕的箱子装入到一个货车的车厢中，为了提高物流效率，需要将车厢尽量填满，显然，车厢如果能被 $100\%$ 填满是最优的，但通常认为，车厢能够填满 $85\%$，认为装箱是优化的。

设车厢为长方形，其长宽高分别为 $L,W,H$；共有 $n$ 个箱子，箱子也为长方形，第 $i$ 个箱子的长宽高为 $l_i$, $w_i$, $h_i$（ $n$ 个箱子的体积总和是要远远大于车厢的体积），做以下假设和要求：

1. 长方形的车厢共有 8 个角，并设靠近驾驶室并位于下端的一个角的坐标为 $(0,0,0)$ ，车厢共6个面，其中长的4个面，以及靠近驾驶室的面是封闭的，只有一个面是开着的，用于工人搬运箱子

2. 需要计算出每个箱子在车厢中的坐标，即每个箱子摆放后，其和车厢坐标为 $(0,0,0)$ 的角相对应的角在车厢中的坐标，并计算车厢的填充率。

### 2. 基于体积的贪心策略求解

初始的箱子排序是基于其体积从大到小排序。将箱子按照给定策略放入容器中，并返回最终的空间利用率，步骤如下：
1. 对箱子进行初始排序，假设给定箱子序列 $(b_1,b_2,...,b_n)$，第一个箱子的可放置点是 $(0,0,0)$。
2. 按顺序考虑箱子，主要考虑两条参考线形成的参考面， $z$ 轴上的参考线 $L_z$ 和 $x$ 轴上的 $L_x$。在考虑箱子 $b_i$ 时，先把可放置点按 $y$ 坐标从小到大排序， $y$ 坐标相同的按 $x$ 坐标从小到大排序， $x,y$ 坐标都相同的按 $z$ 坐标从小到大排序，按照排好序的可放位置去检测 $b_i$能否放入该位置。在评判箱子 $b_i$ 能否放入位置 $(x,y,z)$ 中时，不仅要求其不能与车厢和其他箱子相交，要求 $z+h_i\leq L_z, x+l_i\leq L_x$ 。

    在检测一个可放位置位置时，我们尝试所有可放置方向，当找到第 $1$ 个可放入点时，则把箱子 $b_i$ 放入该位置，并更新可放置点。若所有可放置点都不能放入该箱子，则分两种情况考虑：
    1. 若 $L_x< L$，则提高 $x$ 轴上的参考线，把该箱子作为水平方向参考的箱子。
    2. 否则，提高 $z$ 轴上的参考线。如果提高参考线后，该箱子还是不能放下，在此装填中，该箱子不能放入容器中，以后的装填中将不再考虑该箱子。

2. 逐个尝试将箱子放入容器中，根据不同的姿态选择最佳放置方式。若第一个箱子可以放入点 $(0,0,0)$，则第 $2$ 个箱子的可放置点有 $3$ 个，分别为 $(l_1,0,0)$, $(0,w_1,0)$, $(0,0,h_1)$
3. 假设第 $2$ 个箱子选择了点 $(l_1,0,0)$，则我们删除点 $(l_1,0,0)$，同时增加点 $(l_1+l_2,0,0)$, $(l_1,w_2,0)$, $(l_1,0,h_2)$
3. 第 $3$ 个箱子的可放置点有 $5$ 个，以此类推。
4. 考虑第 $i$ 个箱子，若其选择了点 $(x,y,z)$，则我们先从可放置点中删除点 $(x,y,z)$，再增加 $(x+l_i,y,z), (z,y+w_i,z), (x,y,z+h_1)$。
5. 若所有可放置点都不能放入第 $i$ 个箱子，则放置点不更新，直接考虑第 $i+1$ 个箱子。记录已放置的箱子，并返回总的体积利用率。

```python
def encase_cargos_into_container(cargos:Iterable, container:Container, strategy:type) -> float:
    sorted_cargos:List[Cargo] = strategy.encasement_sequence(cargos)
    i = 0 # 记录当前货物
    while i < len(sorted_cargos):
        j = 0 # 记录当前摆放方式
        cargo = sorted_cargos[i]
        poses = strategy.choose_cargo_poses(cargo, container)

        temp_flag = []
        while j < len(poses):
            cargo.pose = poses[j]
            is_encased = container._encase(cargo)
            temp_flag.append(deepcopy(is_encased))
            if is_encased.is_valid:
                break # 可以装入 不再考虑后续摆放方式
            j += 1  # 不可装入 查看下一个摆放方式
        container._extend_points(cargo, temp_flag[-1])
        
        if is_encased.is_valid:
            container._setted_cargos.append(cargo)
            i += 1 # 成功放入 继续装箱
        elif is_encased == Point(-1,-1,0):
            continue # 没放进去但是修改了参考面位置 重装
        else:
            i += 1 # 纯纯没放进去 跳过看下一个箱子
    return sum(list(map(lambda cargo:cargo.volume,container._setted_cargos))) / container.volume
```

### 2.1 类定义和方法

#### `CargoPose` 和 `Point`

用于定义箱子的姿态和位置。
- **`is_encasable`**：检查一个箱子是否可以放入指定位置。
- **`_encase`**：尝试将箱子放入容器中。
- **`_extend_points`**：更新可用放置点。
- **`_adjust_setting_cargo`**：调整箱子的位置以减少空间浪费。
- **`rectangles_overlap_area_sum`**：计算与三个平面的投影重叠面积和。
- **`rectangles_overlap_area_bottom`**：计算与底面投影重叠面积。

```python
from enum import Enum

class CargoPose(Enum):
    tall_wide = 0
    tall_thin = 1
    mid_wide = 2
    mid_thin = 3
    short_wide = 4
    short_thin = 5

class Point(object):
    def __init__(self, x: int, y: int, z: int) -> None:
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self) -> str:
        return f"({self.x},{self.y},{self.z})"
    
    def __eq__(self, __o: object) -> bool:
        return self.x == __o.x and self.y == __o.y and self.z == __o.z

    @property
    def is_valid(self) -> bool:
        return self.x >= 0 and self.y >= 0 and self.z >= 0
    
    @property
    def tuple(self) -> tuple:
        return (self.x, self.y, self.z)
```

- `CargoPose`：定义箱子可能的姿态。
- `Point`：用于表示三维空间中的位置。

#### `Cargo`

表示一个箱子及其属性。

```python
class Cargo(object):
    def __init__(self, length: int, width: int, height: int) -> None:
        self._point = Point(-1, -1, -1)
        self._shape = {length, width, height}
        self._pose = CargoPose.tall_thin

    def __repr__(self) -> str:
        return f"{self._point} {self.shape}"

    @property
    def pose(self) -> CargoPose:
        return self._pose

    @pose.setter
    def pose(self, new_pose: CargoPose):
        self._pose = new_pose

    @property
    def _shape_swiche(self) -> dict:
        edges = sorted(self._shape)
        return {
            CargoPose.tall_thin: (edges[1], edges[0], edges[-1]),
            CargoPose.tall_wide: (edges[0], edges[1], edges[-1]),
            CargoPose.mid_thin: (edges[-1], edges[0], edges[1]),
            CargoPose.mid_wide: (edges[0], edges[-1], edges[-1]),
            CargoPose.short_thin: (edges[-1], edges[1], edges[0]),
            CargoPose.short_wide: (edges[1], edges[-1], edges[0])
        }

    @property
    def point(self):
        return self._point

    @point.setter
    def point(self, new_point:Point):
        self._point = new_point

    @property
    def shape(self) -> tuple:
        return self._shape_swiche[self._pose]

    @property
    def volume(self) -> int:
        reslut = 1
        for i in self._shape:
            reslut *= i
        return reslut
```

- 包含属性如`point`、`shape`和`volume`。
- `pose`决定了当前箱子的放置姿态。

#### `Container`

表示容器及其方法。

```python
class Container(object):
    def __init__(self, length: int, width: int, height: int) -> None:
        self._length = length
        self._width = width
        self._height = height
        self._refresh()

    def _refresh(self):
        self._horizontal_planar = 0  # Place the reference surface horizontally
        self._vertical_planar = 0  # Position reference plane vertically
        self._available_points = [Point(0, 0, 0)]  # Ordered collection of placeable points
        self._setted_cargos: List[Cargo] = []
        self.plane_shadow = [[], [], []] # Create a cargo list that adheres to the coordinate plane

    def is_encasable(self, site: Point, cargo: Cargo) -> bool:
        encasable = True
        temp = deepcopy(cargo)
        temp.point = site
        if (
            temp.x + temp.length > self.length or
            temp.y + temp.width > self.width or
            temp.z + temp.height > self.height
        ):
            encasable = False
        for setted_cargo in self._setted_cargos:
            if _is_cargos_collide(temp, setted_cargo):
                encasable = False
        return encasable

    def _encase(self, cargo: Cargo) -> Point:
        # The flag stores the placement position, (-1, -1, 0) placement fails and the reference surface is adjusted, (-1, -1, -1) placement fails.
        flag = Point(-1, -1, -1)  
        # Used to record the reference plane position before execution to facilitate subsequent comparison.
        history = [self._horizontal_planar, self._vertical_planar]
        def __is_planar_changed() -> bool:
            return (
                not flag.is_valid and # To prevent damage to the points that have been determined to be placed, that is, they can only be changed on the basis of (-1, -1, -1)
                self._horizontal_planar == history[0] and 
                self._vertical_planar == history[-1]
            ) 
        for point in self._available_points:
            if (
                self.is_encasable(point, cargo) and
                point.x + cargo.length < self._horizontal_planar and
                point.z + cargo.height < self._vertical_planar
            ):
                flag = point
                break
        if not flag.is_valid:
            if (
                self._horizontal_planar == 0 or
                self._horizontal_planar == self.length
            ):
                if self.is_encasable(Point(0, 0, self._vertical_planar), cargo):
                    flag = Point(0, 0, self._vertical_planar)
                    self._vertical_planar += cargo.height
                    self._horizontal_planar = cargo.length 
                elif self._vertical_planar < self.height:
                    self._vertical_planar = self.height
                    self._horizontal_planar = self.length
                    if __is_planar_changed():
                        flag.z == 0
            else:
                for point in self._available_points:
                    if (
                        point.x == self._horizontal_planar and
                        point.y == 0 and
                        self.is_encasable(point, cargo) and
                        point.z + cargo.height <= self._vertical_planar
                    ):
                        flag = point
                        self._horizontal_planar += cargo.length
                        break
                if not flag.is_valid:
                    self._horizontal_planar = self.length
                    if __is_planar_changed():
                        flag.z == 0
        return flag

    def _extend_points(self, cargo, flag):
        if flag.is_valid:
            cargo.point = flag
            if flag in self._available_points:
                self._available_points.remove(flag)
            self._adjust_setting_cargo(cargo)
            self._available_points.extend([
                Point(cargo.x + cargo.length, cargo.y, cargo.z),
                Point(cargo.x, cargo.y + cargo.width, cargo.z),
                Point(cargo.x, cargo.y, cargo.z + cargo.height)
            ])
            if cargo.x == 0:
                self.plane_shadow[0].append(cargo)
            if cargo.y == 0:
                self.plane_shadow[1].append(cargo)
            if cargo.z == 0:
                self.plane_shadow[2].append(cargo)
            self._sort_available_points()

    def _adjust_setting_cargo(self, cargo: Cargo):
        site = cargo.point
        temp = deepcopy(cargo)
        if not self.is_encasable(site, cargo):
            return None
        xyz = [site.x, site.y, site.z] 
        for i in range(3): 
            is_continue = True
            while xyz[i] > 1 and is_continue:
                xyz[i] -= 1
                temp.point = Point(xyz[0], xyz[1], xyz[2])
                for setted_cargo in self._setted_cargos:
                    if not _is_cargos_collide(setted_cargo, temp):
                        continue
                    xyz[i] += 1
                    is_continue = False
                    break
        cargo.point = Point(xyz[0], xyz[1], xyz[2])
    
    def rectangles_overlap_area_sum(self, cargo: Cargo):
        temp = deepcopy(cargo)
        area = 0
        for i in range(3):
            for j in range(len(self.plane_shadow[i])):
                plusarea = rectangles_overlap_area(self.plane_shadow[i][j], temp, i)
                area += plusarea
        return area

    def rectangles_overlap_area_bottom(self, cargo: Cargo):
        temp = deepcopy(cargo)
        area = 0
        for j in range(len(self.plane_shadow[2])):
            plusarea = rectangles_overlap_area(self.plane_shadow[2][j], temp, 2)
            area += plusarea
        return area

    def save_encasement_as_file(self):
        file = open(f"{int(time())}_encasement.csv",'w',encoding='utf-8')
        file.write(f"index,x,y,z,length,width,height\n")
        i = 1
        for cargo in self._setted_cargos:
            file.write(f"{i},{cargo.x},{cargo.y},{cargo.z},")
            file.write(f"{cargo.length},{cargo.width},{cargo.height}\n")
            i += 1
        file.write(f"container,,,,{self}\n")
        file.close()

    @property
    def length(self) -> int:
        return self._length

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def volume(self) -> int:
        return self.height * self.length * self.width
```

### 2.2 碰撞检测

与其他箱子的碰撞检测：直接在三维空间中确定两个实体是否冲突困难。在这里，我们采用的方法是在二维平面上考虑三维空间中的实体投影，在 $xy$ 面、 $xz$ 面和 $yz$ 面上的投影情况判断是否冲突。

任意平行于长方形车厢放置的长方体货物，如果它们在任意三个方向上的投影没有重叠，则两者就没有冲突。而对于平面上的长方体而言，相对在左边的长方体的右上角坐标如果小于相对右边的长方体的左下角坐标，

```python
def _is_rectangles_overlap(rec1:tuple, rec2:tuple) -> bool:
    return not (
        rec1[0] >= rec2[2] or rec1[1] >= rec2[3] or
        rec2[0] >= rec1[2] or rec2[1] >= rec1[3]
    )

def rectangles_overlap_area(cargo0:Cargo, cargo1:Cargo, opt):
    area = 0
    if opt == 0:
        rec0 = cargo0.get_shadow_of("yz")
        rec1 = cargo1.get_shadow_of("yz")
        if _is_rectangles_overlap(rec0, rec1):
            area += min(abs(rec0[0]-rec1[2]), abs(rec0[2]-rec1[0])) * min(abs(rec0[1]-rec1[3]), abs(rec0[3]-rec1[1]))
    if opt == 1:
        rec0 = cargo0.get_shadow_of("xz")
        rec1 = cargo1.get_shadow_of("xz")
        if _is_rectangles_overlap(rec0, rec1):
            area += min(abs(rec0[0]-rec1[2]), abs(rec0[2]-rec1[0])) * min(abs(rec0[1]-rec1[3]), abs(rec0[3]-rec1[1])) 
    if opt == 2:
        rec0 = cargo0.get_shadow_of("yz")
        rec1 = cargo1.get_shadow_of("yz")
        if _is_rectangles_overlap(rec0, rec1):
            area += min(abs(rec0[0]-rec1[2]), abs(rec0[2]-rec1[0])) * min(abs(rec0[1]-rec1[3]), abs(rec0[3]-rec1[1]))   
    return area

def _is_cargos_collide(cargo0: Cargo, cargo1: Cargo) -> bool:
    return (
        _is_rectangles_overlap(cargo0.get_shadow_of("xy"), cargo1.get_shadow_of("xy")) and
        _is_rectangles_overlap(cargo0.get_shadow_of("yz"), cargo1.get_shadow_of("yz")) and
        _is_rectangles_overlap(cargo0.get_shadow_of("xz"), cargo1.get_shadow_of("xz"))
    )
```

- **`_is_rectangles_overlap`**：检查两个矩形是否重叠。
- **`rectangles_overlap_area`**：计算投影重叠面积。
- **`_is_cargos_collide`**：检查两个箱子是否碰撞。

### 3. 退火算法求解

在初始算法基础上，加入模拟退火过程进行优化：

- **模拟退火的参数**：
  - `St`：初始温度
  - `Et`：结束温度

```python
if __name__ == "__main__":
    # Simulated annealing parameter settings
    St = 1
    L = 0
    Et = 0.4
    dL = 3
    dt = 0.9

    # E1-1
    cargos = [Cargo(108, 76, 30 ) for _ in range(40)]
    cargos.extend([Cargo(110, 43, 25) for _ in range(33)])
    cargos.extend([Cargo(92, 81, 55) for _ in range(39)])

    case = Container(587,233,220)
    sorted_list = sorted(cargos, key=lambda cargo: cargo.volume, reverse=1)
    start_list = deepcopy(sorted_list)
    fstart = encase_cargos_into_container(cargos, case, sorted_cargos=sorted_list) # Space utilization during initial placement
    f = fstart
    fbest = fstart
    Bbest = case._setted_cargos # The initial placement order
    for i in [1, 2]: # Perform annealing twice
        t = St
        Lt = L
        while t >= Et:
            for j in range(Lt):
                cargos = [Cargo(108, 76, 30) for _ in range(40)]
                cargos.extend([Cargo(110, 43, 25) for _ in range(33)])
                cargos.extend([Cargo(92, 81, 55) for _ in range(39)])
                case = Container(587, 233, 220)

                # Randomly swap the order of two nodes in a path
                s1, s2 = randint(0, int(len(sorted_list)/2) - 1), randint(int(len(sorted_list)/2), len(sorted_list) - 1)
                start_list[s1], start_list[s2] = start_list[s2], start_list[s1]

                temp_list = deepcopy(start_list)
                temp_list2 = deepcopy(start_list)
                f1 = encase_cargos_into_container(cargos, case, sorted_cargos=temp_list)
                B1 = case._setted_cargos
                df = f1 - f
                if df > 0:
                    f = f1
                    B = B1
                    if f > fbest:
                        fbest = f
                        Bbest = B
                        start_list = temp_list2
                else:
                    x = random()
                    if x < np.exp(10*df/t):
                        f = f1
                        B = B1
                        start_list = temp_list2
            Lt += dL
            t *= dt

    print("initial rate of capacity=", fstart)
    print("highest rate of capacity=", fbest)
```
