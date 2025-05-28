import pickle
import numpy as np
import src.utils as u
import matplotlib.pyplot as plt
instance_file = './data/ch150.txt'
C, data, T = u.readInstance(instance_file)
def compute_path_length_from_matrix(tour, distance_matrix):
    """
    使用距离矩阵 C 计算路径总长度
    tour: [0, 3, 5, 0]
    distance_matrix: C[i][j] 表示城市 i 到城市 j 的距离
    """
    length = 0.0
    for i in range(len(tour) - 1):
        length += distance_matrix[tour[i], tour[i + 1]]
    return length

def plot_solution_tours(city_coordinates, tours, title="Solution", save_path=None):
    plt.figure(figsize=(8, 8))
    colors = ['r', 'g', 'b', 'm', 'c', 'orange', 'purple', 'brown']
    x = [p[0] for p in city_coordinates]
    y = [p[1] for p in city_coordinates]
    plt.scatter(x, y, c='blue', label='Cities')
    plt.plot(city_coordinates[0][0], city_coordinates[0][1], 'or', label='Depot (0)')
    for i, tour in enumerate(tours):
        path = [city_coordinates[c] for c in tour]
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        plt.plot(xs, ys, color=colors[i % len(colors)], label=f'Salesman {i+1}')
    plt.title(title)
    plt.grid()
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def extract_tours(tour_matrix):
    """
    将路径矩阵转换为城市访问顺序路径（每个销售员一条路径）
    输入：
        tour_matrix: (m, n, n) 路径矩阵
    输出：
        list of list: [[0, 3, 5, 1, 0], [...], ...]
    """
    tours = []
    for t in tour_matrix:
        path = []
        current = 0
        visited = set([0])
        while True:
            next_city = np.argmax(t[current])
            if next_city == 0 or next_city in visited:
                break
            path.append(next_city)
            visited.add(next_city)
            current = next_city
        tours.append([0] + path + [0])  # 起点终点都为城市 0（depot）
    return tours

f = open('./Results/data/julei.pkl','rb')
data = pickle.load(f)
from src.utils import readInstance
_, coords, _ = readInstance('./data/ch150.txt')  # 如果是从文件读取的
sorted_front = sorted(data[0], key=lambda ind: -ind.function_vals[0])  # 因为 function_vals[0] 是负数
for j, ind in enumerate(sorted_front):
    print(f'solution{j+1}')
    print("Objective Values:", ind.function_vals)
    matrix = u.getTourMatrix(ind)
    tours = extract_tours(matrix)
    for i, tour in enumerate(tours):
        length = compute_path_length_from_matrix(tour, C)
        print(f"Salesman {i + 1}: {tour} | Length: {length}")
    if j in [0, len(data[0]) // 2, len(data[0]) - 1]:
        title = f"Solution {j + 1} | Objectives: {ind.function_vals}"
        save_path = f"Results/path/julei_{j + 1}.png"
        plot_solution_tours(coords, tours, title, save_path)