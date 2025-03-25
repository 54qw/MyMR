import copy

def crowdedComparisonOperator(i,j):     #根据非支配等级和拥挤距离比较两个染色体优先级
    if i.nondomination < j.nondomination or (i.nondomination == j.nondomination and
                                                    i.crowding_distance > j.crowding_distance):
        return -1
    elif j.nondomination < i.nondomination or (i.nondomination == j.nondomination and
                                                    j.crowding_distance > i.crowding_distance):
        return 1
    else:
        return 0

def dominates(individual,other):    #判断个体是否支配其它
    value = False
    if (individual.function_vals[0] > other.function_vals[0]) or (
                individual.function_vals[1] > other.function_vals[1]):
        if (individual.function_vals[0] >= other.function_vals[0]) and (
                individual.function_vals[1] >= other.function_vals[1]):
            value = True
    return value

def nondominatedSort(population):     #实现当前种群非支配排序，形成 Pareto 前沿
    nondominated_fronts = [[]]
    for individual in population:
        individual.dominated_solutions = []
        individual.domination_count = 0
        for other in population:
            if dominates(individual,other):
                individual.dominated_solutions.append(other)
            elif dominates(other,individual):
                individual.domination_count += 1
        if individual.domination_count == 0:
            individual.nondomination = 1
            nondominated_fronts[0].append(individual)

    i = 0
    while len(nondominated_fronts[i]) > 0:
        Q = []
        for individual in nondominated_fronts[i]:
            for dominated in individual.dominated_solutions:
                dominated.domination_count -= 1
                if dominated.domination_count == 0:
                    dominated.nondomination = i + 2
                    Q.append(dominated)
        i += 1
        nondominated_fronts.append(Q)
    return nondominated_fronts

def assignCrowdingDistance(list_of_individuals):    #拥挤距离计算
    list_I = copy.deepcopy(list_of_individuals)
    l = len(list_I)
    for i in list_I:
        i.crowding_distance = 0
    n_objs = len(list_I[0].function_vals)
    for i in range(n_objs):
        list_I.sort(key=lambda x: x.function_vals[i])
        fmax = list_I[-1].function_vals[i]
        fmin = list_I[0].function_vals[i]
        list_I[0].crowding_distance = 10**9
        list_I[-1].crowding_distance = 10**9
        for j in range(1, l-1):
            list_I[j].crowding_distance += (list_I[j + 1].function_vals[i] -
                                            list_I[j - 1].function_vals[i]) / (
                                                   fmax - fmin)
    return list_I

























