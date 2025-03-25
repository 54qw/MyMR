import numpy as np

#读取数据
def read_tsp(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    assert 'NODE_COORD_SECTION\n' in lines
    index = lines.index('NODE_COORD_SECTION\n')
    data = lines[index + 1:-1]
    tmp = []
    for line in data:
        line = line.strip().split(' ')
        if line[0] == 'EOF':
            continue
        tmpline = []
        for x in line:
            if x == '':
                continue
            else:
                tmpline.append(float(x))
        if tmpline == []:
            continue
        tmp.append(tmpline)
    data = tmp
    data = np.array(data)
    num = data.shape[0]
    return data,num

#计算不同点之间的距离，生成矩阵
def compute_dis_mat(num_node,location):
    dis_mat = np.zeros((num_node,num_node))
    for i in range(num_node):
        for j in range(num_node):
            if i == j:
                dis_mat[i][j] = np.inf
                continue
            a = location[i][1:]
            b = location[j][1:]
            tmp = np.sqrt(sum([(x[0] - x[1]) ** 2 for x in zip(a, b)]))
            dis_mat[i][j] = tmp
    return dis_mat


if __name__ == '__main__':
    data, num = read_tsp('./data/ch150.tsp')
    dis_mat = compute_dis_mat(num,data)
    print(dis_mat)