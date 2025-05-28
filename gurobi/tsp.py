from gurobipy import *
import numpy as np

model = Model("question1")

# 读取城市坐标
coord = []
with open("data.txt", "r") as lines:
    lines = lines.readlines()
for line in lines:
    xy = line.split()
    coord.append(xy)
coord = np.array(coord)
w, h = coord.shape
coordinates = np.zeros((w, h), float)
for i in range(w):
    for j in range(h):
        coordinates[i, j] = float(coord[i, j])

# x、y坐标
data_x = coordinates[:, 0]
data_y = coordinates[:, 1]

data_num = len(data_x)
distance_juzheng = np.zeros((data_num, data_num))

for i in range(data_num):
    for j in range(data_num):
        if i == j:
            distance_juzheng[i, j] = 1e6  # 避免自己到自己
        else:
            distance_juzheng[i, j] = np.hypot(data_x[i] - data_x[j], data_y[i] - data_y[j])  # 更简洁！

# 定义决策变量
x = model.addVars(data_num, data_num, vtype=GRB.BINARY)
u = model.addVars(data_num, vtype=GRB.CONTINUOUS)

# 构造目标函数
model.setObjective(quicksum(x[i, j] * distance_juzheng[i, j] for i in range(data_num) for j in range(data_num)), GRB.MINIMIZE)

# 每个城市出发一次
for i in range(data_num):
    model.addConstr(quicksum(x[i, j] for j in range(data_num)) == 1)

# 每个城市到达一次
for j in range(data_num):
    model.addConstr(quicksum(x[i, j] for i in range(data_num)) == 1)

# 子环约束（MTZ约束）
for i in range(1, data_num):
    for j in range(1, data_num):
        if i != j:
            model.addConstr(u[i] - u[j] + data_num * x[i, j] <= data_num - 1)

model.optimize()

# 输出结果
if model.status == GRB.OPTIMAL:
    best = np.zeros((data_num, data_num), dtype=int)
    for i in range(data_num):
        for j in range(data_num):
            if x[i, j].X > 0.5:
                best[i, j] = 1
    print(f"最优目标值为：{model.ObjVal}")
    print(f"最优分配为：\n{best}")
else:
    print("未找到最优解")
