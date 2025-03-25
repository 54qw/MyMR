import numpy as np
import copy
import functools
from . import nsga2
from . import chromosome as chrom

def tournamentSelection(population, tsub=2, selection_probability=1):     #锦标赛选择
    rng = np.random.default_rng()
    parents = []
    probability_list = [selection_probability]
    best = None
    for i in range(1,tsub):
        probability_list.append(selection_probability*
                                (1-selection_probability)**i)
    while len(parents)<len(population):
        tournament_bracket = rng.choice(population,tsub,replace=False).tolist()
        tournament_bracket = sorted(tournament_bracket,
                                    key=functools.cmp_to_key(
                                        nsga2.crowdedComparisonOperator))
        for i in range(tsub):
            best = tournament_bracket[tsub-1]
            if rng.choice([0,1],
                          p=[probability_list[i],1-probability_list[i]])==0:
                if parents==[] or parents[-1]!=tournament_bracket[i]:
                    best = tournament_bracket[i]
                    break
        parents.append(best)
    return parents

def partiallyMappedCrossover(p1, p2):  # 部分交叉映射（PMX）
    rng = np.random.default_rng()
    p11 = p1.part1
    p21 = p2.part1
    p12 = p1.part2
    p22 = p2.part2
    child_1 = chrom.Chromosome()
    child_1.part1 = np.zeros(shape=p11.shape, dtype=int)
    child_1.part2 = np.zeros(shape=p12.shape, dtype=int)
    child_2 = chrom.Chromosome()
    child_2.part1 = np.zeros(shape=p21.shape, dtype=int)
    child_2.part2 = np.zeros(shape=p22.shape, dtype=int)
    cut_points = np.sort(rng.choice(p11.shape[0], 2, replace=False))
    for i in range(cut_points[0], cut_points[1]):
        child_2.part1[i], child_1.part1[i] = p11[i], p21[i]
    for i in range(p11.shape[0]):
        if p11[i] not in child_1.part1 and child_1.part1[i] == 0:
            child_1.part1[i] = p11[i]
        if p21[i] not in child_2.part1 and child_2.part1[i] == 0:
            child_2.part1[i] = p21[i]
    for i in range(p11.shape[0]):
        if child_1.part1[i] == 0:
            child_1.part1[i] = rng.choice([j for j in p11 if j
                                            not in child_1.part1])
        if child_2.part1[i] == 0:
            child_2.part1[i] = rng.choice([j for j in p21 if j
                                            not in child_2.part1])
    child_1.part2 = np.sort(rng.choice(np.arange(1, max(p11)), p12.shape[0],
                                        replace=False))
    child_2.part2 = np.sort(rng.choice(np.arange(1, max(p21)), p22.shape[0],
                                        replace=False))

    return (child_1, child_2)

def getCxChild(a,c,child_a,start_id):   #循环交叉
    count = 0
    i = copy.deepcopy(start_id)
    child_a = np.zeros(shape=a.shape,dtype=int)
    while True:
        child_a[i] = a[i]
        k = copy.deepcopy(i)
        i = next(j for j in range(a.shape[0]) if c[k]==a[j])
        count += 1
        if count==a.shape[0]:
            return child_a
        if c[k]==a[start_id]:
            for j in range(a.shape[0]):
                if child_a[j]==0:
                    child_a[j] = c[j]
            return child_a

def cyclicCrossover(p1, p2):
    rng = np.random.default_rng()
    p11 = p1.part1
    p21 = p2.part1
    p12 = p1.part2
    p22 = p2.part2
    child_1 = chrom.Chromosome()
    child_1.part1 = np.empty(shape=p11.shape, dtype=int)
    child_1.part2 = np.empty(shape=p12.shape, dtype=int)
    child_2 = chrom.Chromosome()
    child_2.part1 = np.empty(shape=p21.shape, dtype=int)
    child_2.part2 = np.empty(shape=p22.shape, dtype=int)
    start_id = 0
    child_1.part1 = getCxChild(p11, p21, child_1.part1, start_id)
    child_2.part1 = getCxChild(p21, p11, child_2.part1, start_id)
    child_1.part2 = np.sort(rng.choice(np.arange(1, max(p11)), p12.shape[0],
                                        replace=False))
    child_2.part2 = np.sort(rng.choice(np.arange(1, max(p21)), p22.shape[0],
                                        replace=False))

    return (child_1, child_2)

def orderedCrossover(p1, p2):  # 顺序交叉
    rng = np.random.default_rng()
    p11 = p1.part1
    p21 = p2.part1
    p12 = p1.part2
    p22 = p2.part2
    child_1 = chrom.Chromosome()
    child_1.part1 = np.zeros(shape=p11.shape, dtype=int)
    child_1.part2 = np.zeros(shape=p12.shape, dtype=int)
    child_2 = chrom.Chromosome()
    child_2.part1 = np.zeros(shape=p21.shape, dtype=int)
    child_2.part2 = np.zeros(shape=p22.shape, dtype=int)
    cut_points = np.sort(rng.choice(p11.shape[0], 2, replace=False))
    for i in range(cut_points[0], cut_points[1]):
        child_1.part1[i], child_2.part1[i] = p11[i], p21[i]
    remnant_ids = np.concatenate((np.arange(cut_points[1], p11.shape[0]),
                                  np.arange(cut_points[0])))
    cut_ids = np.concatenate((np.arange(cut_points[1], p11.shape[0]),
                              np.arange(cut_points[1])))
    rearr_p11 = [p11[i] for i in cut_ids]
    rearr_p21 = [p21[i] for i in cut_ids]
    rem_p11 = [i for i in rearr_p11 if i not in child_2.part1]
    rem_p21 = [i for i in rearr_p21 if i not in child_1.part1]
    j = 0
    for i in remnant_ids:
        child_2.part1[i], child_1.part1[i] = rem_p11[j], rem_p21[j]
        j += 1
    child_1.part2 = np.sort(rng.choice(np.arange(1, max(p11)), p12.shape[0],
                                        replace=False))
    child_2.part2 = np.sort(rng.choice(np.arange(1, max(p21)), p22.shape[0],
                                        replace=False))

    return (child_1, child_2)

def decodeChromosome(chromosome):  # 解码
    decoded = [0]
    encoded_part1 = copy.deepcopy(chromosome.part1)
    encoded_part2 = copy.deepcopy(chromosome.part2)
    count = 0
    for i, val in enumerate(encoded_part1):
        if i != encoded_part2[count]:
            decoded.append(val)
        else:
            decoded.append(0)
            decoded.append(val)
            if count != len(encoded_part2) - 1:
                count += 1
    return decoded

def rationalizeHgaResult(org_result):
    if org_result[0] != 0:
        i = org_result.index(0)
        cut_out = org_result[:i]
        cut_out.reverse()
        new_result = [j for j in org_result[i:] if j != 0]
        new_result.extend(cut_out)
    else:
        new_result = [j for j in org_result if j != 0]
    return new_result


def hierarchicalCrossover(p1, p2, C, T, ptype=1):  # HX
    rng = np.random.default_rng()
    child_1 = chrom.Chromosome()
    child_2 = chrom.Chromosome()
    p11 = p1.part1.tolist()
    p21 = p2.part1.tolist()
    dp1 = decodeChromosome(p1)
    dp2 = decodeChromosome(p2)
    k = rng.choice(p11)
    result_1 = [k]
    while len(p11) > 1:
        i = p11.index(k)
        j = p21.index(k)
        cities = []
        left_city_1, right_city_1 = p11[i - 1], p11[(i + 1) % len(p11)]
        left_city_2, right_city_2 = p21[j - 1], p21[(j + 1) % len(p21)]

        if i == len(p11) - 1:
            cities.append(left_city_1)
        else:
            cities.append(right_city_1)
        if j == len(p21) - 1:
            cities.append(left_city_2)
        else:
            cities.append(right_city_2)
        if ptype == 1:
            distances = [C[k, cities[0]], C[k, cities[1]]]
        else:
            distances = [T[k, cities[0]], T[k, cities[1]]]
        p11.remove(k)
        p21.remove(k)
        k = cities[np.argsort(distances)[0]]
        result_1.append(k)
    k = rng.choice(p1.part1)
    result_2 = [k]
    while len(dp1) > 1:

        i = dp1.index(k)
        j = dp2.index(k)
        cities = []
        left_city_1, right_city_1 = dp1[i - 1], dp1[(i + 1) % len(dp1)]
        left_city_2, right_city_2 = dp2[j - 1], dp2[(j + 1) % len(dp2)]

        if i == len(dp1) - 1:
            cities.append(left_city_1)
        else:
            cities.append(right_city_1)
        if j == len(dp2) - 1:
            cities.append(left_city_2)
        else:
            cities.append(right_city_2)
        dp1.remove(k)
        dp2.remove(k)
        if C[k, cities[0]] > C[k, cities[1]]:
            k = cities[1]
        else:
            k = cities[0]
        result_2.append(k)
    result_2 = rationalizeHgaResult(result_2)
    # child_2.part2 = np.sort(rng.choice(np.arange(1, max(p2.part1)),
    #                                     p2.part2.shape[0], replace=False))
    child_2.part2 = mutate_part2(p2.part2, len(p21))
    child_1.part1 = np.array(result_1)
    child_2.part1 = np.array(result_2)
    if rng.choice([0, 1]) == 0:
        # child_1.part2 = mutate_part2(p1.part2, len(p11))
        child_1.part2 = copy.deepcopy(p2.part2)
    else:
        child_1.part2 = copy.deepcopy(p1.part2)
    return child_1, child_2


def mutate_part2(part2, part1_len):
    rng = np.random.default_rng()
    m = len(part2)
    if m == 0:
        return part2  # 空的直接返回

    keep_num = rng.integers(1, m + 1)  # +1 保证至少留一个
    keep_indices = rng.choice(range(m), size=keep_num, replace=False)
    kept = np.array([part2[i] for i in keep_indices])

    new_needed = m - keep_num
    all_choices = np.setdiff1d(np.arange(2, part1_len), kept)

    if new_needed > 0 and all_choices.size >= new_needed:
        new_vals = rng.choice(all_choices, size=new_needed, replace=False)
        return np.sort(np.concatenate([kept, new_vals]))
    else:
        return np.sort(kept)


def insertMutation(child):  # 插入变异
    rng = np.random.default_rng()
    p1 = child.part1
    p2 = child.part2
    mutated = chrom.Chromosome()
    mutated.part1 = copy.deepcopy(p1)
    # mutated.part2 = np.sort(rng.choice(np.arange(1, max(p1)),p2.shape[0], replace=False))
    if rng.choice([0, 1]) == 0:
        mutated.part2 = mutate_part2(p2, len(p1))
    else:
        mutated.part2 = copy.deepcopy(p2)
    point_1 = rng.choice(np.arange(p1.shape[0] - 1))
    point_2 = rng.choice(np.arange(point_1 + 1, p1.shape[0]))
    mutated.part1 = np.insert(mutated.part1, point_1 + 1, p1[point_2])
    mutated.part1 = np.delete(mutated.part1, point_2 + 1)

    return mutated

def swapMutation(child):  # 交换变异
    rng = np.random.default_rng()
    p1 = child.part1
    p2 = child.part2
    mutated = chrom.Chromosome()
    mutated.part1 = copy.deepcopy(p1)
    # mutated.part2 = np.sort(rng.choice(np.arange(1, max(p1)), p2.shape[0],
    #                                     replace=False))
    if rng.choice([0, 1]) == 0:
        mutated.part2 = mutate_part2(p2, len(p1))
    else:
        mutated.part2 = copy.deepcopy(p2)
    points = rng.choice(np.arange(p1.shape[0]), 2, replace=False)
    mutated.part1[points[0]], mutated.part1[points[1]] = mutated.part1[
        points[1]], mutated.part1[points[0]]

    return mutated

def invertMutation(child):  # 反转变异
    rng = np.random.default_rng()
    p1 = child.part1
    p2 = child.part2
    mutated = chrom.Chromosome()
    mutated.part1 = copy.deepcopy(p1)
    # mutated.part2 = np.sort(rng.choice(np.arange(1, max(p1)), p2.shape[0],
    #                                     replace=False))
    if rng.choice([0, 1]) == 0:
        mutated.part2 = mutate_part2(p2, len(p1))
    else:
        mutated.part2 = copy.deepcopy(p2)
    points = np.sort(rng.choice(np.arange(p1.shape[0]), 2, replace=False))
    inverse_p1 = [p1[i] for i in range(points[1], points[0] - 1, -1)]
    j = 0
    for i in range(points[1], points[0] - 1, -1):
        mutated.part1[i] = inverse_p1[j]
        j += 1

    return mutated

def scrambleMutation(child):  # 打乱变异
    rng = np.random.default_rng()
    p1 = child.part1
    p2 = child.part2
    mutated = chrom.Chromosome()
    mutated.part1 = copy.deepcopy(p1)
    # mutated.part2 = np.sort(rng.choice(np.arange(1, max(p1)), p2.shape[0],
    #                                     replace=False))
    if rng.choice([0, 1]) == 0:
        mutated.part2 = mutate_part2(p2, len(p1))
    else:
        mutated.part2 = copy.deepcopy(p2)
    points = np.sort(rng.choice(np.arange(p1.shape[0]), 2, replace=False))
    scramble_ids = rng.permutation(np.arange(points[0], points[1] + 1))
    scrambled_p1 = [p1[i] for i in scramble_ids]
    for i, scrambled_id in enumerate(scramble_ids):
        mutated.part1[scrambled_id] = scrambled_p1[i]

    return mutated

def mutateChild(child):
    rng = np.random.default_rng()
    mutated = None
    mu_type = rng.choice([0,1,2,3])
    if mu_type==0:
        mutated = insertMutation(child)
    elif mu_type==1:
        mutated = swapMutation(child)
    elif mu_type==2:
        mutated = invertMutation(child)
    else:
        mutated = scrambleMutation(child)
    return mutated