from gurobipy import Model, GRB

n = 4
weights = [2, 3, 4, 5]
values = [3, 4, 5, 6]
capacity = 5

m = Model("knapsack")

# 决策变量：是否选取物品
x = [m.addVar(vtype=GRB.BINARY, name=f"x_{i}") for i in range(n)]

# 约束：重量不能超过capacity
m.addConstr(sum(weights[i]*x[i] for i in range(n)) <= capacity, "weight_limit")

# 目标：最大化总价值
m.setObjective(sum(values[i]*x[i] for i in range(n)), GRB.MAXIMIZE)

m.optimize()

for i in range(n):
    print(f"x[{i}] = {x[i].x}")
print(f"最大价值: {m.objVal}")
