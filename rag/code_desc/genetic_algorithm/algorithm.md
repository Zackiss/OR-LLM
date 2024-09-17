### 1. 要求

物流公司在流通过程中，需要将打包完毕的箱子装入到一个货车的车厢中，为了提高物流效率，需要将车厢尽量填满，显然，车厢如果能被 $100\%$ 填满是最优的，但通常认为，车厢能够填满 $85\%$，认为装箱是优化的。

设车厢为长方形，其长宽高分别为 $L,W,H$；共有 $n$ 个箱子，箱子也为长方形，第 $i$ 个箱子的长宽高为 $l_i$, $w_i$, $h_i$（ $n$ 个箱子的体积总和是要远远大于车厢的体积），做以下假设和要求：

1. 长方形的车厢共有 8 个角，并设靠近驾驶室并位于下端的一个角的坐标为 $(0,0,0)$ ，车厢共6个面，其中长的4个面，以及靠近驾驶室的面是封闭的，只有一个面是开着的，用于工人搬运箱子

2. 需要计算出每个箱子在车厢中的坐标，即每个箱子摆放后，其和车厢坐标为 $(0,0,0)$ 的角相对应的角在车厢中的坐标，并计算车厢的填充率。

### 2. 有偏随机密钥遗传算法求解
算法由两个部分组成，一个是搜索解空间的有偏随机密钥遗传算法，另一个是在遗传算法中评估每个解表示的启发式算法(称为箱子打包过程)。简单地说，该算法将每个解(装箱顺序和箱子摆放方向)用一个序列进行编码，该序列可以通过启发式算法进行评估，使遗传算法能够通过选择找到好的可行解。

该算法本质上是一个衍生的基于扩展随机密钥和从种群中有偏选择个体的遗传算法。在大多数三维装箱的元启发式算法中，它们不是表示放置箱子的每个坐标，而是为寻找最佳打包顺序设计一个基于某种规则的包装过程算法，如最深的底部、左侧、填充方法。在获得高质量解的同时，可以大大简化解的表示。此外，包装序列总可以映射到一个可行的包装方案，而无需担心坐标重叠，这意味着此遗传算法实现中不需要进行染色体修复。

### 2.1 随机密钥表示
随机密钥是一种编解码染色体的方法，其中解表示为 $[0, 1]$ 内的实数向量。假设要打包的箱子数为 $n$。在这个实现中，一个随机密钥的长度总是 $2n$。利用随机数生成随机密钥。
```python
population = np.random.uniform(low=0.0, high=1.0, size=(self.num_individuals, self.num_gene))
```
每个解决方案中，前 $n$ 个基因，为装箱顺序，代表了 $n$ 个要打包箱子的放入容器中的顺序，可以通过按相应基因值的升序排序来解码。可以使用 `argsort()` 方法获取已排序数组的索引。
前 $n$ 个基因定义为货物的摆放顺序，利用 `np.argsort` 函数进行升序排列，之后获取对应的索引。
```python
sorted_indexs = np.argsort(fitness_list)
```
后 $n$ 个基因，为箱子方向矢量，代表了箱子的摆放方向。在三维的设置中，总共有六个方向来放置一个箱子。在某些情况下，一些箱子不能颠倒放置，也不能被垂直方向所限制。要考虑所有可能的方向，每个基因的方向编码称为 `BO`。
```python
# Determine the placement status of goods
def orient(self, box, BO=1):
    d, w, h = box
    # BO indicates the placement status of the goods
    if BO == 1:
        return (d, w, h)
    elif BO == 2:
        return (d, h, w)
    elif BO == 3:
        return (w, d, h)
    elif BO == 4:
        return (w, h, d)
    elif BO == 5:
        return (h, d, w)
    elif BO == 6:
        return (h, w, d)

def selecte_box_orientaion(self, VBO, box, EMS):
    # In each encoding solution, the first n genes are defined as the cargo placement sequence (BPS)
    # The n genetic codes after the random key represent the placement status of the goods (one of 6 types)
    # BOs represents all possible placement states of goods, and the goods placement state vector
    BOs = []
    for direction in [1, 2, 3, 4, 5, 6]:
        if self.fitin(self.orient(box, direction), EMS):
            BOs.append(direction)

    # Select the specific placement state of the goods based on the goods placement state vector.
    selectedBO = BOs[math.ceil(VBO * len(BOs)) - 1]
    return selectedBO
```
综上所述，有偏随机密钥遗传算法用一个在 $[0, 1]$ 之间的实数向量来表示一个解，它的长度为 $2n$，由长度为 $n$ 的装箱顺序和长度为 $n$ 的箱子方向矢量组成。有箱子包装过程，就可对该解决方案的质量效果评估。

### 2.2 选择个体
有偏随机密钥遗传算法和其他遗传算法之间的区别是，在每一代种群基于适应度值被划分为两个组，精英组和非精英组。这种有偏差的选择将极大地影响遗传算法中的操作，如交叉操作和种群进化。定义一个函数，根据适应度值和精英个体的数量将种群划分为精英组和非精英组。
```python
# Each generation population is divided into elite population and non-elite population according to the fitness function value.
def partition(self, population, fitness_list):
    # The first n genes are defined as the order in which the goods are placed. Use the np.argsort function to arrange them in ascending order, and then obtain the corresponding index.
    sorted_indexs = np.argsort(fitness_list)
    return population[sorted_indexs[:self.num_elites]], population[sorted_indexs[self.num_elites:]], \
    np.array(fitness_list)[sorted_indexs[:self.num_elites]]
```
### 2.3 交叉操作
在该算法中，对于每一次杂交，父母中总有一个来自精英组中而另一个来自非精英组。然后，它产生一个后代的基因是来自精英组或非精英组的基因，由基于预先确定的概率确定。一般来说，这种概率设置更有利于从精英组中继承基因。杂交后代的数目取决于每个种群中的个体数和突变体的数量。
```python
def crossover(self, elite, non_elite):
    # Crossover operation, for each gene, there is a certain probability of selecting from the elite group or the non-elite group
    return [elite[gene] if np.random.uniform(low=0.0, high=1.0) < self.eliteCProb else non_elite[gene] for gene in
            range(self.num_gene)]
```
### 2.4 变异操作
没有执行变异操作(例如交换变异)，而是在新一代中创建了随机生成的新个体，以此作为增加种群随机性的手段。确定种群变异个体的数目，定义一个函数来创建新的个体，就像初始化种群。
```python
# Mutation operation does not produce mutant genes based on the original genes, but creates new individuals to be inserted into the population.
def mutants(self):
    return np.random.uniform(low=0.0, high=1.0, size=(self.num_mutants, self.num_gene))
```
### 2.5 进化操作
对于每一代，所有的精英组个体都被保留给下一个种群，不做任何修改。此外，变异产生的新个体直接添加到下一个种群中。由于这个问题是关于最小化使用的容器的数量，我们将在每一代更新最小适应值。我们可以将进化过程的函数定义为：
```python
# Biased selection of parents, including one from the elite group and the other from the non-elite group
def mating(self, elites, non_elites):
    # The number of offspring depends on the number of individuals in each population and the number of mutants
    num_offspring = self.num_individuals - self.num_elites - self.num_mutants
    return [self.crossover(random.choice(elites), random.choice(non_elites)) for i in range(num_offspring)]
    # Evolution process: All individuals in the elite group are retained in the next generation population, and new individuals generated during crossover and mutation are also retained in the next generation population. The fitness function of each generation is evaluated and the optimal value is retained.
def fit(self, patient=15, verbose=True):
    # Initialize the population information and encode it. The population information is expressed as a vector within the range of [0,1]
    population = np.random.uniform(low=0.0, high=1.0, size=(self.num_individuals, self.num_gene))
    # Initialize fitness function value
    fitness_list = self.cal_fitness(population)
    # Repeat genetic iteration
    best_iter = 0
    for g in range(self.num_generations):
        # If the current population and the optimal population meet certain conditions, terminate the iteration early and the search ends
        if g - best_iter > patient:
            self.used_bins = math.floor(best_fitness)
            self.best_fitness = best_fitness
            self.solution = best_solution
            if verbose:
                print('Stop iteration.', g)
            return 'Solution found.'

        # Select elite group, non-elite group, and fitness function
        elites, non_elites, elite_fitness_list = self.partition(population, fitness_list)

        # Cross over in a biased manner to produce new individuals in offspring
        offsprings = self.mating(elites, non_elites)

        # Mutation produces new individuals in future generations
        mutants = self.mutants()

        # Generate all new offspring and their fitness function values
        offspring = np.concatenate((mutants, offsprings), axis=0)

        offspring_fitness_list = self.cal_fitness(offspring)
        # The new population consists of all elite groups and produces new offspring
        population = np.concatenate((elites, offspring), axis=0)
        fitness_list = list(elite_fitness_list) + list(offspring_fitness_list)

        # Update the optimal fitness function value
        for fitness in fitness_list:
            if fitness < best_fitness:
                best_iter = g
                best_fitness = fitness
                best_solution = population[np.argmin(fitness_list)]
```
如何评估一个解决方案的适应值（用函数计算适应度值)。一个解决方案告诉箱子打包顺序和摆放方向。为了评估每个解决方案的适应度函数，只需严格按照解决方案中提供的信息来打包这些箱子，然后计算使用了多少个容器。因此，需要一个按照解决方案将三维箱子包装到固定尺寸的容器。这个解决方案称为放置箱子程序。

### 2.6 最大空间表示法：剩余可用空间
首先，需要一个函数来反映箱子和容器的三维空间，同时判定重叠和越界。其次，必须定义一个函数来放置箱子。
```python
# Place a cargo in a remaining available space
boxToPlace = np.array(box)
# Select a remaining free space among the remaining free spaces
selected_min = np.array(selected_EMS[0])
# Calculate the maximum space left by the cargo in the remaining available space selected
ems = [selected_min, selected_min + boxToPlace]
self.load_items.append(ems)
```
最大空间是一个概念来表示一个矩形空间的最低限度和最大坐标，它只有在物体垂直于三维的情况下才起作用，适用于三维装箱问题。
通过记录容器中剩余的可用空间来记录箱子的放置情况的最大空间，可将容器中的盒子放置看作以下过程：
1. 在一个剩余可用空间中放置一个货物。
2. 生成新的剩余可用空间，此由货物之间的交叉点产生，在此步可以计算新的剩余空间。
3. 消除掉完全被其他剩余可用空间包含的新产生的剩余可用空间。
4. 新产生的剩余可用空间不能小于剩余货物的体积。
5. 新产生的剩余可用空间尺寸不能小于剩余货物中最小的尺寸。
6. 如果有效，则添加新的剩余可用空间。

注意，在三维空间中有六个由交点产生的空间。然而，在实践中将盒子最小坐标相对于选定的扩展空间，因为两个盒子之间间隙通常导致不适合其他盒子的间隙。
```python
# Minimum and maximum coordinates of the remaining available space for the intersection
x1, y1, z1 = EMS[0]
x2, y2, z2 = EMS[1]
# Minimum and maximum coordinates of the cargo
# x3, y3, z3 = ems[0]
x4, y4, z4 = ems[1]
# Generate new remaining available space in three-dimensional space
# new_EMSs = [
#     [(x1, y1, z1), (x3, y2, z2)],
#     [(x4, y1, z1), (x2, y2, z2)],
#     [(x1, y1, z1), (x2, y3, z2)],
#     [(x1, y4, z1), (x2, y2, z2)],
#     [(x1, y1, z1), (x2, y2, z3)],
#     [(x1, y1, z4), (x2, y2, z2)]
# ]
# In practice, the minimum coordinate of the box is equal to the remaining available space, because usually the gap left between two items is difficult to place new items
new_EMSs = [
    [np.array((x4, y1, z1)), np.array((x2, y2, z2))],
    [np.array((x1, y4, z1)), np.array((x2, y2, z2))],
    [np.array((x1, y1, z4)), np.array((x2, y2, z2))]
]
```
若要检查一个扩展空间重叠或完全被另一个扩展空间包含，可以定义以下函数来计算两个条件：
```python
# Check if one remaining free space overlaps another remaining free space
def overlapped(self, ems, EMS):
    if np.all(ems[1] > EMS[0]) and np.all(ems[0] < EMS[1]):
        return True
    return False

# Checks whether one remaining free space is included in another's remaining free space
def inscribed(self, ems, EMS):
    if np.all(EMS[0] <= ems[0]) and np.all(ems[1] <= EMS[1]):
        return True
    return False
```
现在，有了一种方法来更新容器和盒子的状态，以及在每个盒子放置后的三维空间。接下来，介绍一个放置箱子启发式规则，以决定在一个箱子放置在哪个剩余空间中。

### 2.7 启发式规则：到右上角距离最大化
最深最左填充是一个打包一系列箱子的启发式规则，在这个规则中，它总是选择最小坐标的空间放置箱子。启发式算法的目标是在每次迭代中将盒子放置在最深的空间，希望最终所有的盒子都能紧密地放置在一起。但是一些最优解不能由这种启发式规则构造。为了解决这个问题，改进了放置箱子的启发式规则，规则命名为到容器右上角的距离最大化。该启发式算法总是将盒子放置在可用剩余可用空间中，使得盒子与容器的右上角的距离最大化。
```python
# Heuristic rule: front top angle distance from container (truck)
def DFTRC_2(self, box, k):
    maxDist = -1
    selectedEMS = None

    for EMS in self.Bins[k].EMSs:
        # D, W, H are the depth, width and height of a container (truck)
        D, W, H = self.Bins[k].dimensions
        # Traverse the placement status of goods
        for direction in [1, 2, 3, 4, 5, 6]:
            d, w, h = self.orient(box, direction)
            # If the goods satisfy the current remaining available space
            if self.fitin((d, w, h), EMS):
                # Minimum coordinates of remaining available space
                x, y, z = EMS[0]
                # Distance between remaining available space and container (truck)
                distance = pow(D - x - d, 2) + pow(W - y - w, 2) + pow(H - z - h, 2)
                # Find remaining free space at maximum distance
                if distance > maxDist:
                    maxDist = distance
                    selectedEMS = EMS
    return selectedEMS
```
该启发式规则用于选择一个扩展空间从现有的剩余可用空间用于放置箱子。如果箱子不能放置在现有的空间，将打开一个新的空容器，并恢复正在进行的放置过程。

所有箱子根据以上算法装入容器中，接下来需要定义一个函数计算解决方案的适应度值。可以将使用过的容器数作为解决方案的适应度值。但是如果两个解决方案使用的容器数相同，它们将得到相同的适应度值。因此需要将适应度值进行一个小的调整，附加了最小装填容器的正则项来修正。这种计算适应度值的基本原理是，如果两个解决方案使用的容器数相同，在未装满的容器中装载量较少的容器，在其他容器中箱子摆放的更加紧凑。因此，提升改进以此得到更好解的潜力更大。

利用使用集装箱的数量进行计算适应度函数，如果两个方案使用了相同的数量的集装箱，以及产生了相同的适应度函数的值，在没有装满的货物里面如果货物数量最少，在满的的集装箱里是摆放的更加紧凑：
```python
def evaluate(self):
    if self.infisible:
        return INFEASIBLE

    leastLoad = 1
    for k in range(self.num_opend_bins):
        load = self.Bins[k].load()
        if load < leastLoad:
            leastLoad = load
    return self.num_opend_bins + leastLoad % 1
```
