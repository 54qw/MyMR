import os
import sys
import copy
import numpy as np
import matplotlib.pyplot as plt
import functools
import pickle
from tkinter import Tk,filedialog
from . import nsga2

def getTourMatrix(chromosome):  # 染色体解码为路径三维邻接矩阵
    part_1 = chromosome.part1
    part_2 = chromosome.part2
    n_tours = part_2.shape[0] + 1
    n_cities = part_1.shape[0]
    tour_matrix = np.zeros(shape=(n_tours, n_cities+1, n_cities+1), dtype=int)
    tour_order_lists = [[] for i in range(n_tours)]
    count = 0
    breakpoints = [part_2[i] for i in range(n_tours-1)]
    breakpoints.append(n_cities)
    for i in range(n_tours):
        breakpoint = breakpoints[i]
        for j in range(count, breakpoint):
            tour_order_lists[i].append(part_1[j])
        count = copy.deepcopy(breakpoint)

    for i, tours in enumerate(tour_order_lists):
        for j, city in enumerate(tours):
            if j != 0:
                tour_matrix[i, tours[j-1], city] = 1
            else:
                tour_matrix[i, 0, city] = 1
            if j == len(tours) - 1:
                tour_matrix[i, city, 0] = 1
    return tour_matrix

def euclideanDistance(p1, p2):
    return ((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)**0.5

def readInstance(filename): # 读取城市坐标数据
    city_coordinates = []
    with open(filename) as f:
        for line in f.readlines():
            if line[0].isnumeric():
                int_list = [int(float(i)) for i in line.split()]
                city_coordinates.append((int_list[1], int_list[2]))
    distance_matrix = np.empty(shape=(len(city_coordinates), len(city_coordinates)), dtype=int)
    for i, city_i in enumerate(city_coordinates):
        for j, city_j in enumerate(city_coordinates):
            distance_matrix[i, j] = round(euclideanDistance(city_i, city_j))

    plt.figure()
    x = [p[0] for p in city_coordinates]
    y = [p[1] for p in city_coordinates]
    plt.plot(x, y, 'ob')
    plt.plot(city_coordinates[0][0], city_coordinates[0][1], 'or')
    plt.grid()
    plt.show()

    rng = np.random.default_rng()
    time_matrix = np.empty(shape=(len(city_coordinates),
                                      len(city_coordinates)),dtype=float)
    for i,city_i in enumerate(city_coordinates):
        for j,city_j in enumerate(city_coordinates):
            time_matrix[i,j] = distance_matrix[i,j]/rng.integers(low=20000,high=90001)
    return distance_matrix,city_coordinates,time_matrix

def plotAndSaveFigures(best_front, pop_size, ptype=1, saverno='n'):
    plt.figure()
    plt.xlabel('Total cost (distance)')
    if ptype == 1:
        plt.ylabel('Max-min tour distances (amplitude)')
    else:
        plt.ylabel('Sum(individual tour time-avg. tour time)')
    final_front = nsga2.nondominatedSort(best_front)
    bestest_front = nsga2.assignCrowdingDistance(final_front[0])
    bestest_front = sorted(bestest_front, key=functools.cmp_to_key(nsga2.crowdedComparisonOperator))
    fvalues1 = [(i.function_vals[0], i.function_vals[1])
                for i in bestest_front[:pop_size]]
    X = [-i[0] for i in fvalues1]
    Y = [-i[1] for i in fvalues1]
    plt.plot(X, Y, 'og')

    if saverno == 'y':
        script_dir = os.getcwd()
        results_dir_2 = os.path.join(script_dir, "Results/figures/")
        if not os.path.isdir(results_dir_2):
            os.makedirs(results_dir_2)
        plt.savefig(results_dir_2 + "julei.png")
        plt.savefig(results_dir_2 + "julei.svg")

    plt.show()


def saveData(fronts):
    script_dir = os.getcwd()
    results_dir_1 = os.path.join(script_dir, "Results/data/")
    if not os.path.isdir(results_dir_1):
        os.makedirs(results_dir_1)
    with open(results_dir_1 + "julei.pkl", "wb") as wrfile:
        pickle.dump(fronts, wrfile)


























