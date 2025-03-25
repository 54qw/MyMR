import numpy as np
from . import utils
from . import mtsp

class Chromosome(object):
    def __init__(self):
        self.part1 = None
        self.part2 = None
        self.domination_count = None
        self.nondomination = None
        self.dominated_solutions = None
        self.crowding_distance = None
        self.function_vals = None
    def __eq__(self,other):
        if isinstance(self, other.__class__):
            return ((np.all(self.part1 == other.part1)) and (np.all(self.part2 == other.part2)))
        return False

def applyConstraintPenalty(individual, C, max_tour_length=3500):
    """
    对于每个旅行商路径超过 max_tour_length 的，加入惩罚
    """
    X = utils.getTourMatrix(individual)

    penalty = 0
    for tour in X:
        length = np.sum(np.multiply(C, tour))
        if length > max_tour_length:
            penalty += (length - max_tour_length)

    # 加权惩罚（你可以调整惩罚系数）
    penalty_weight = 10000  # 惩罚因子
    total_penalty = penalty_weight * penalty

    # 修改目标函数值（注意：目标是最大化 -cost，所以我们加负惩罚）
    individual.function_vals[0] -= total_penalty

def loadCitiesFromTxt(path):
    data = np.loadtxt(path)
    coords = data[:, 1:]  # 取第2、3列作为坐标
    return coords  # shape=(n, 2)

def sectorBasedClusteringFromCoords(coords, n_tours):
    depot = coords[0]
    cities = coords[1:]
    vectors = cities - depot
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])  # [-pi, pi]
    angles = (angles + 2 * np.pi) % (2 * np.pi)  # [0, 2pi)

    min_angle = np.min(angles)
    max_angle = np.max(angles)
    angle_range = max_angle - min_angle

    if angle_range == 0:
        return np.zeros(len(cities), dtype=int)  # 所有角度相同

    sector_size = angle_range / n_tours
    labels = ((angles - min_angle) // sector_size).astype(int)
    labels = np.clip(labels, 0, n_tours - 1)  # 修正溢出标签
    return labels

def creatChromosome1(C, T, n_tours, ptype=1):
    rng = np.random.default_rng()
    n_cities = C.shape[0]

    chromosome = Chromosome()
    chromosome.part1 = rng.permutation(np.arange(1, n_cities))
    chromosome.part2 = np.sort(rng.choice(np.arange(1,n_cities-1), n_tours-1, replace=False))
    X = utils.getTourMatrix(chromosome)
    a = mtsp.objectiveFunction1(C, X)
    b = mtsp.objectiveFunction2(C, T, X, ptype)
    chromosome.function_vals = [a, b]
    applyConstraintPenalty(chromosome, C, max_tour_length=3500)
    return chromosome

def creatChromosome2(C, T, n_tours, ptype=1):
    path = './data/ch150.txt'
    rng = np.random.default_rng()
    n_cities = C.shape[0]
    coords = loadCitiesFromTxt(path)
    depot = coords[0]
    city_coords = coords[1:]
    labels = sectorBasedClusteringFromCoords(coords, n_tours)

    part1 = []
    part2_list = []
    for cid in range(n_tours):
        city_idx = np.where(labels == cid)[0]
        city_ids = city_idx + 1  # 对应 coords[1:] => 原城市编号从1开始

        # 可选：簇内按距离 depot 排序
        dists = np.linalg.norm(city_coords[city_idx] - depot, axis=1)
        sorted_ids = city_ids[np.argsort(dists)]
        if rng.random() < 0.8:
            rng.shuffle(sorted_ids)
        part1.extend(sorted_ids.tolist())
        part2_list.append(len(sorted_ids))
    part2 = np.cumsum(part2_list)[:-1]
    perturbed = []
    for val in part2:
        offset = rng.integers(-5, 5)  # -1, 0, 1
        perturbed_val = min(max(val + offset, 1), len(part1) - 1)
        perturbed.append(perturbed_val)
    chromosome = Chromosome()
    chromosome.part1 = np.array(part1)
    chromosome.part2 = np.sort(np.array(perturbed))
    X = utils.getTourMatrix(chromosome)
    a = mtsp.objectiveFunction1(C, X)
    b = mtsp.objectiveFunction2(C, T, X, ptype)
    chromosome.function_vals = [a, b]
    applyConstraintPenalty(chromosome, C, max_tour_length=3500)
    return chromosome

# def creatInitialPopulation(N, C, T, data, n_tours, ptype=1):
#     population = []
#     while len(population) < N:
#         individual = creatChromosome(C, T, n_tours, ptype)
#         if individual not in population:
#             population.append(individual)
#     return population

def creatInitialPopulation(N, C, T, data, n_tours, ptype=1):
    population = []
    clustered_count = int(N * 0.5)

    while len(population) < clustered_count:
        ind = creatChromosome2(C, T, n_tours, ptype)
        if ind not in population:
            population.append(ind)

    while len(population) < N:
        ind = creatChromosome1(C, T, n_tours, ptype)
        if ind not in population:
            population.append(ind)

    return population


if __name__ == '__main__':
    # txt_path = '../data/ch150.txt'  # 替换为你自己的文件路径
    # coords = loadCitiesFromTxt(txt_path)
    # labels = sectorBasedClusteringFromCoords(coords, 5)
    # visualize_clusters(coords, labels)
    p1, p2 = creatChromosome2(0,0,n_tours=5)
    print(p1)
    print(p2)