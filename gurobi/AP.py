import numpy as np
from gurobipy import Model, GRB

n = 3
cost = np.array([
    [10, 2, 6],
    [3, 8, 5],
    [4, 7, 9]
])

m = Model("assignment")

# x[i,j]=1 表示工人i做任务j
x = {}
for i in range(n):
    for j in range(n):
        x[i,j] = m.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")

# 每个工人只能做一个任务
for i in range(n):
    m.addConstr(sum(x[i,j] for j in range(n)) == 1)

# 每个任务只能被一个工人完成
for j in range(n):
    m.addConstr(sum(x[i,j] for i in range(n)) == 1)

# 目标：最小总成本
m.setObjective(sum(cost[i,j]*x[i,j] for i in range(n) for j in range(n)), GRB.MINIMIZE)

m.optimize()

for i in range(n):
    for j in range(n):
        if x[i,j].x > 0.5:
            print(f"工人{i} -> 任务{j}")
print(f"总成本: {m.objVal}")
