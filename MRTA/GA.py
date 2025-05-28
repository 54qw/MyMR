import math
import random
import numpy as np
from matplotlib import pyplot as plt
import utils

class GA:
    def __init__(self, robot_num, start_index, filename, max_gen, distance_weight, balance_weight,):
        self.population_size = 100   #种群规模大小
        self.population = []    #种群
        self.robot_num = robot_num  #机器人数量
        self.start_index = start_index  #出发点序号取值[1，点数]
        self.max_gen = max_gen  #最大迭代次数

        self.cross_prob = 0.8  # 交叉概率
        self.mutation_prob = 0.15  # 变异概率
        self.cross_pmx_prob = 0.5  # 交叉选择部分匹配交叉PMX的概率，这部分没用到，只用到cross_ox_prob
        self.cross_ox_prob = 0.5  # 交叉选择顺序匹配交叉OX的概率

        self.mutation_swap_prob = 0.3  # 变异选择"交换两个元素"的概率
        self.mutation_reverse_prob = 0.4  # 变异选择"反转两个元素之间所有元素"的概率
        self.mutation_insert_prob = 1 - self.mutation_swap_prob - self.mutation_reverse_prob  # 变异选择"一个元素插入到另一个元素后面"的概率

        self.distance_weight = distance_weight  #总路程权重
        self.balance_weight = balance_weight    #均衡度权重

        self.filename = filename
        self.nodes_data, self.node_num = utils.read_tsp(self.filename)
        self.chrom_len = self.robot_num + self.node_num - 2  # 染色体长度=（巡检点数量-1）+（机器人数量-1），由编码方式决定
        self.dummy_points = [x for x in range(self.node_num + 1, self.node_num + self.robot_num)]  # 虚点
        self.dis_mat = utils.compute_dis_mat(self.node_num, self.nodes_data.copy())

        self.cur_pop_best_chrom = []  # 当代的最优个体
        self.cur_pop_best_chrom_fit = 0  # 当代的最优个体的适应度
        self.cur_pop_best_path = []  # 当代最优个体的路线
        self.cur_pop_best_dis_sum = np.Inf  # 当代最优个体的总距离
        self.all_cur_pop_best_chrom = []  # 记录每次迭代过程中当代最优个体变化情况
        self.all_cur_pop_best_chrom_fit = []  # 记录每次迭代过程中当代最优个体的适应度变化情况（可以不必设置，因为check_vertex_fitness_func可以计算单体适应度）
        self.all_cur_pop_best_dist_sum = []  # 记录每次迭代过程中当代最优个体的总距离变化情况

        self.best_chrom = []  # 全局最优个体，不一定是每代的最优个体，每代的最优个体可能比以往的最优个体差
        self.best_chrom_fit = 0  # 全局最优个体的适应度
        self.best_path = []  # 全局最优个体的路径
        self.best_dis_sum = np.Inf  # 全局最优个体的路径之和
        self.all_best_chrom = []  # 记录每次迭代过程中全局最优个体的变化情况
        self.all_best_chrom_fit = []  # 记录每次迭代过程中全局最优个体的适应度变化情况
        self.all_best_dist_sum = []  # 记录每次迭代过程中全局最优个体的总距离

    def random_init_pop(self):
        """
                chrom组成
                如：总共8个点，3号点为起始点.下列染色体组成我们把起点和终点省略，看情况是否增加虚点
                1个旅行商：一个chrom = [把3剔除，其余数字由1到8组成]
                    如[1,5,4,2,6,8,7]表示旅行商路线为3->1->5->4->2->6->8->7->3
                2个旅行商：一个chrom = [1个9(9代表虚点，其实也是起点3)，其余数字由1到8组成]。以此类推到多个旅行商的情况。
                    如[1,5,4,9,2,6,8,7]表示：
                        旅行商1路线为3->1->5->4->3(9)
                        旅行商2路线为3(9)->2->6->8->7->3
                3个旅行商：一个chrom = [9,10，其余数字由1到8组成]
                    如[1,5,4,9,2,6,10,8,7]
                        旅行商1路线为3->1->5->4->3(9)
                        旅行商2路线为3->2->6->3(10)
                        旅行商3路线为3->8->7->3
        """
        for i in range(self.population_size):
            chrom = [x for x in range(1, self.node_num+1)]
            chrom.remove(self.start_index)  #剔除起始点
            chrom.extend(self.dummy_points) #添加虚点
            random.shuffle(chrom)
            self.population.append(chrom)
        #初始化全局最优个体及其适应度
        self.best_chrom = self.population[0]
        self.best_chrom_fit = self.fitness_func(self.best_chrom)

    def get_real_routes(self,chrom):
        """
        Args:
            chrom: 染色体

        Returns:
            all_routes：每条染色体代表的所有巡检路线组成列表
            routes_dis：每条路线总路程组成列表
        """
        tmp_chrom = chrom[:]
        # 将增加的虚点还原成起始点
        for i in range(len(chrom)):
            if chrom[i] in self.dummy_points:
                tmp_chrom[i] = self.start_index
        # 根据起始点把chrom分成多段
        one_route = []
        all_routes = [] # 所有巡检路线
        for x in tmp_chrom:
            if x == self.start_index:
                all_routes.append(one_route)
                one_route = []
            else:
                one_route.append(x)
        all_routes.append(one_route)
        all_routes = [[self.start_index] + route + [self.start_index] for route in all_routes]

        routes_dis = [] # 每条路线总距离组成的列表
        # 获取各点之间的距离矩阵
        # dis_mat = utils.compute_dis_mat(self.node_num, self.nodes_data.copy())
        for r in all_routes:
            distance = 0
            if len(r) <= 2: # 一个机器人不出门
                distance = 999999
                routes_dis.append(distance)
            else:
                r_len = len(r)
                for i in range(r_len-1):
                    distance += self.dis_mat[r[i]-1][r[i+1]-1]
                routes_dis.append(distance)
        return all_routes, routes_dis

    def fitness_func(self, chrom):
        """
                计算个体的适应度值，即个体目标函数值的倒数
                计算个体的目标函数值
                目标函数 Z = distance_weight*总路程 + balance_weight*均衡度
                均衡度 = (max(l)-min(l))/ max(l)
                Args:
                    chrom:染色体

                Returns:
                    个体的适应度
        """
        all_routes, routes_dis = self.get_real_routes(chrom)
        sum_dis = sum(routes_dis)
        max_dis = max(routes_dis)
        min_dis = min(routes_dis)
        if max_dis == 0:
            balance = 0
        elif min_dis == 0:
            balance = np.Inf
        else:
            balance = (max_dis - min_dis) / max_dis
        obj = self.distance_weight * sum_dis + self.balance_weight * balance
        # fitness = math.exp(1.0 / obj)
        fitness = 1 / max_dis
        return fitness

    def compute_pop_fitness(self, population):
        """
        计算当前种群所有个体的的适应度
        Args:
            population: 种群

        Returns:
            种群所有个体的的适应度
        """
        return [self.fitness_func(chrom) for chrom in population]

    def get_best_chrom(self,population):
        """
        找到种群中最优个体
        Args:
            population: 种群

        Returns:
            population[index]：最优个体
            index:最优个体在种群中下标
        """
        tmp = self.compute_pop_fitness(population)
        index = tmp.index(max(tmp))
        return population[index], index

    def binary_tournament_select(self, population):
        """
                二元锦标赛：从种群中抽取2个个体参与竞争，获胜者个体进入到下一代种群
                Args:
                    population: 目前种群

                Returns:
                    new_population:下一代种群
        """
        pop = population.copy()
        new_population = []
        for i in range(len(pop)//2):
            competitors = random.choices(pop,k=2)
            winner = max(competitors, key=lambda x: self.fitness_func(x))
            new_population.append(winner)
            pop.remove(winner)
        return new_population

    def ga_choose(self, population):
        population_fitness = self.compute_pop_fitness(population)
        sum_fitness = sum(population_fitness)
        fitness_ratio = [sub*1.0/sum_fitness for sub in population_fitness]
        rand1 = np.random.rand()
        rand2 = np.random.rand()
        for i, sub in enumerate(fitness_ratio):
            if rand1 >= 0:
                rand1 -= sub
                if rand1 < 0:
                    index1 = i
            if rand2 >= 0:
                rand2 -= sub
                if rand2 < 0:
                    index2 = i
            if rand1 < 0 and rand2 < 0:
                break
        return list(population[index1]), list(population[index2])

    def cross_ox(self,parent_chrom1, parent_chrom2):
        """
        将parent_chrom1被选中的基因片段复制给child_chrom1，
        这里复制是指child_chrom1[index1:index2] = parent_chrom1[index1:index2]
        然后parent_chrom2除了temp_gene1包含的基因，parent_chrom2剩下基因按照顺序放到child_chrom1中
        即(|parent_chrom2|-|temp_gene1|)基因按顺序放到child_chrom1。
        同理，child_chrom2交换一下parent_chrom1和parent_chrom2，也可以得到
        如：
        parent_chrom1 = [1, 2, 3, 4, 5, 6, 7, 8, 9], sel1选中部分[3, 4, 5, 6]
        parent_chrom2 = [5, 7, 4, 9, 1, 3, 6, 2, 8], sel2选中部分[6, 9, 2, 1]
        child_chrom1  = [7, 9, 3, 4, 5, 6, 1, 2, 8]
            1、child_chrom1对应部分放入sel1
            2、遍历parent_chrom2，parent_chrom2不属于sel1部分基因，按照顺序放入
        """
        # print('A')
        index1, index2 = random.randint(0, self.chrom_len-1), random.randint(0, self.chrom_len-1)
        # print('B')
        if index1 > index2:
            index1, index2 = index2, index1
        temp_gene1 = parent_chrom1[index1:index2]
        temp_gene2 = parent_chrom2[index1:index2]
        child_chrom1, child_chrom2 = [], []
        child_p1, child_p2 = 0, 0
        for i in parent_chrom2:
            if child_p1 == index1:
                child_chrom1.extend(temp_gene1)
                child_p1 += 1
            if i not in temp_gene1:
                child_chrom1.append(i)
                child_p1 += 1
        for i in parent_chrom1:
            if child_p2 == index1:
                child_chrom2.extend(temp_gene2)
                child_p2 += 1
            if i not in temp_gene2:
                child_chrom2.append(i)
                child_p2 += 1
        # print('C')
        return child_chrom1, child_chrom2

    def cross_pmx(self,parent_chrom1, parent_chrom2):
        """
        如：
        index1 = 2, index2 = 6
        parent_chrom1 = [1, 2, 3, 4, 5, 6, 7, 8, 9], 选中部分[3, 4, 5, 6]
        parent_chrom2 = [5, 4, 6, 9, 2, 1, 7, 8, 3], 选中部分[6, 9, 2, 1]
        选中部分的映射关系即1<->6<->3 ; 2<->5 ; 9<->4
        可以看出存在1<->6<->3，说明6在父代1和2选中部分，6后续不需要冲突检测，所以应该1<->3
        """
        # print('D')
        # 随机选择交叉点
        index1, index2 = random.randint(0, self.chrom_len - 1), random.randint(0, self.chrom_len - 1)
        # print('E')
        if index1 > index2:
            index1, index2 = index2, index1
        parent_part1, parent_part2 = parent_chrom1[index1:index2], parent_chrom2[index1:index2]

        child_chrom1, child_chrom2 = [], []
        child_p1, child_p2 = 0, 0  # 指针用来解决复制到指定位置问题
        # 子代1
        for i in parent_chrom1:
            # 指针到达父代的选中部分
            if index1 <= child_p1 < index2:
                # 将父代2选中基因片段复制到子代1指定位置上
                child_chrom1.append(parent_part2[child_p1 - index1])
                child_p1 += 1
                continue
            # 指针未到达父代的选中部分
            if child_p1 < index1 or child_p1 >= index2:
                # 父代1未选中部分含有父代2选中部分基因
                if i in parent_part2:
                    tmp = parent_part1[parent_part2.index(i)]
                    while tmp in parent_part2:
                        tmp = parent_part1[parent_part2.index(tmp)]
                    child_chrom1.append(tmp)
                elif i not in parent_part2:
                    child_chrom1.append(i)
                child_p1 += 1
        # 子代2
        for i in parent_chrom2:
            # 指针到达父代的选中部分
            if index1 <= child_p2 < index2:
                # 将父代1选中基因片段复制到子代2指定位置上
                child_chrom2.append(parent_part1[child_p2 - index1])
                child_p2 += 1
                continue
            # 指针未到达父代的选中部分
            if child_p2 < index1 or child_p2 >= index2:
                # 父代2未选中部分含有父代1选中部分基因
                if i in parent_part1:
                    tmp = parent_part2[parent_part1.index(i)]
                    # 解决1<->6<->3
                    while tmp in parent_part1:
                        tmp = parent_part2[parent_part1.index(tmp)]
                    child_chrom2.append(tmp)
                elif i not in parent_part1:
                    child_chrom2.append(i)
                child_p2 += 1
        # print('F')
        return child_chrom1, child_chrom2

    def crossover(self, parent_chrom1, parent_chrom2):
        """
        种群按概率执行交叉操作
        Args:
            parent_chrom1, parent_chrom2

        Returns:
            child_chrom1, child_chrom2
        """
        prob = np.random.rand()
        # print('a')
        if prob <= self.cross_ox_prob:
            child_chrom1, child_chrom2 = self.cross_ox(parent_chrom1, parent_chrom2)
        else:
            child_chrom1, child_chrom2 = self.cross_pmx(parent_chrom1, parent_chrom2)
        # print('b')
        return child_chrom1, child_chrom2

    def mutate_swap(self, parent_chrom):
        """
        交换变异：当前染色体 [1,5,4,2,6,8,7]，交换1和5位置上元素变成了[1,8,4,2,6,5,7]
        Args:
            parent_chrom: 父代染色体

        Returns:
            child_chrom：交换变异产生的子代染色体
        """
        index1, index2 = random.randint(0, self.chrom_len - 1), random.randint(0, self.chrom_len - 1)
        child_chrom = parent_chrom[:]
        child_chrom[index1], child_chrom[index2] = child_chrom[index2], child_chrom[index1]
        return child_chrom

    def mutate_reverse(self,parent_chrom):
        """
        逆转变异：随机选择两点(可能为同一点)，逆转其中所有的元素
        parent_chrom = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        child_chrom  = [1, 2, 6, 5, 4, 3, 7, 8, 9]
        Args:
            parent_chrom:父代

        Returns:
            child_chrom：逆转变异后的子代
        """
        index1, index2 = random.randint(0, self.chrom_len - 1), random.randint(0, self.chrom_len - 1)
        if index1 > index2:
            index1, index2 = index2, index1
        child_chrom = parent_chrom[:]
        tmp = child_chrom[index1:index2]
        tmp.reverse()
        child_chrom[index1:index2] = tmp
        return child_chrom

    def mutate_insert(self,parent_chrom):
        """
        插入变异：随机选择两个位置，然后将这第二个位置上的元素插入到第一个元素后面。
        parent_chrom = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        child_chrom  = [1, 2, 4, 5, 3, 6, 7, 8, 9]
        Args:
            parent_chrom:父代

        Returns:
            child_chrom：子代
        """
        index1, index2 = random.randint(0, self.chrom_len - 1), random.randint(0, self.chrom_len - 1)
        child_chrom = parent_chrom[:]
        if index1 == index2 or index1 + 1 == index2:
            return child_chrom
        child_chrom.pop(index2)
        child_chrom.insert(index1+1,parent_chrom[index2])
        return child_chrom

    def mutation(self,parent_chrom):
        """
        种群按概率执行变异操作
        Args:
            parent_chrom

        Returns:
            child_chrom
        """
        prob_sum = []
        prob_sum.extend([self.mutation_swap_prob, self.mutation_swap_prob + self.mutation_reverse_prob, 1])
        p = random.random()
        if p <= prob_sum[0]:
            # 交换变异
            child_chrom = self.mutate_swap(parent_chrom)
        elif p <= prob_sum[1]:
            # 逆序变异
            child_chrom = self.mutate_reverse(parent_chrom)
        else:
            # 插入变异
            child_chrom = self.mutate_insert(parent_chrom)
        return child_chrom

    def ga_run(self):
        self.random_init_pop()
        best_dist_list = [] # 全局最优解每一条路径长度
        for i in range(1,self.max_gen):
            parents = self.binary_tournament_select(self.population)
            parents = self.binary_tournament_select(parents)
            fruits = parents.copy()
            num = 1
            while len(fruits) < self.population_size:
                # 选择
                parent_chrom1, parent_chrom2 = self.ga_choose(fruits)
                # print(1)
                # 交叉
                if np.random.rand() < self.cross_prob:
                    child_chrom1, child_chrom2 = self.crossover(parent_chrom1, parent_chrom2)
                # print(2)
                # 变异
                if np.random.rand() < self.mutation_prob:
                    child_chrom1 = self.mutation(child_chrom1)
                if np.random.rand() < self.mutation_prob:
                    child_chrom2 = self.mutation(child_chrom2)
                # print(3)
                fitness1 = self.fitness_func(child_chrom1)
                fitness2 = self.fitness_func(child_chrom2)
                # print(4)
                if fitness1 > fitness2 and child_chrom1 not in fruits:
                    fruits.append(child_chrom1)
                elif fitness2 >= fitness1 and child_chrom2 not in fruits:
                    fruits.append(child_chrom2)
                # print(num)
                # num += 1
            pop_new = fruits

            # 新一代有关参数更新
            # 计算种群所有个体的适应度
            pop_fitness_list = self.compute_pop_fitness(pop_new)
            # 当代最优个体cur_pop_best_chrom及其在种群中的下标best_index
            self.cur_pop_best_chrom, best_index = self.get_best_chrom(pop_new)
            # 当代最优个体的适应度
            self.cur_pop_best_chrom_fit = pop_fitness_list[best_index]
            # 当代最优个体最好的路径组成和每条路路径长度per_pop_best_dist_list
            self.cur_pop_best_path, cur_pop_best_dist_list = self.get_real_routes(self.cur_pop_best_chrom)
            # 当代最优个体所有旅行商路线之和
            self.cur_pop_best_dis_sum = sum(cur_pop_best_dist_list)

            # 记录下当代最优个体
            self.all_cur_pop_best_chrom.append(self.cur_pop_best_chrom)
            # 记录下当代最优个体的适应度
            self.all_cur_pop_best_chrom_fit.append(self.cur_pop_best_chrom_fit)
            # 记录每次迭代过程中当代最优个体的总距离变化情况
            self.all_cur_pop_best_dist_sum.append(self.cur_pop_best_dis_sum)

            # 全局最优个体有关参数更新
            # 当代最优个体与全局最优个体根据适应度比较，如果当代最优个体适应度更大，则更新全局最优个体
            if self.cur_pop_best_chrom_fit > self.best_chrom_fit:
                self.best_chrom = self.cur_pop_best_chrom
                self.best_chrom_fit = self.cur_pop_best_chrom_fit

                self.best_path, best_dist_list = self.get_real_routes(self.best_chrom)
                self.best_dis_sum = self.cur_pop_best_dis_sum

            # 记录下每次迭代过程中全局最优个体
            self.all_best_chrom.append(self.best_chrom)
            # 记录每次迭代过程中全局最优个体的适应度变化情况
            self.all_best_chrom_fit.append(self.best_chrom_fit)
            # 记录每次迭代过程中全局最优个体的总距离
            self.all_best_dist_sum.append(self.best_dis_sum)

            if i % 50 == 0:
            #if self.gen_count >= 0:
                print("经过%d次迭代" % i)
                print("全局最优解距离为：%f，全局最优解长度为%d" % (self.best_dis_sum, len(self.best_chrom)))
                print("全局最优解为{}".format(self.best_chrom))
                print("全局最优解路线为{}".format(self.best_path))
                print("全局最优解路线长度列表为{}".format(best_dist_list))
                print("---------------------------------------------------------")
                print("当代最优解距离为：%f，当代最优解长度为%d" % (self.cur_pop_best_dis_sum, len(self.cur_pop_best_chrom)))
                print("当代最优解为{}".format(self.cur_pop_best_chrom))
                print("当代最优解路线为{}".format(self.cur_pop_best_path))
                print("当代最优解路线长度列表为{}".format(cur_pop_best_dist_list))
                print("**************************************************************************")

            # 更新种群
            self.population = pop_new

    def plot_routes(self):
        plt.figure(figsize=(8, 8))
        colors = ["red", "blue", "green", "purple", "orange", "cyan"]
        # 绘制城市点
        plt.scatter(self.nodes_data[:, 1], self.nodes_data[:, 2], c="black", marker="o", label="巡检点")
        # 标注城市编号
        for i, node in enumerate(self.nodes_data):
            plt.text(node[1], node[2], f"{i+1}", fontsize=9, ha="right")
        # 绘制每个旅行商的路径
        for i, route in enumerate(self.best_path):
            route = [x - 1 for x in route]
            path_coords = self.nodes_data[route]
            color = colors[i % len(colors)]
            plt.plot(
                path_coords[:, 1], path_coords[:, 2],
                marker="o", linestyle="-", color=color,
                label=f"旅行商 {i + 1}"
            )

        # 标注起点
        plt.scatter(self.nodes_data[0][1], self.nodes_data[0][2], c="red", s=100, label="起点")
        title = '路线'
        plt.title(title)
        plt.xlabel("X 坐标")
        plt.ylabel("Y 坐标")
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == '__main__':
    ga_obj = GA(robot_num=5, start_index=1, filename='./data/ch150.tsp',
                max_gen=2000, distance_weight=1, balance_weight=3000)
    ga_obj.ga_run()
    ga_obj.plot_routes()
























