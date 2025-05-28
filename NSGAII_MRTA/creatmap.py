import numpy as np
import matplotlib.pyplot as plt
import os

def create_structured_obstacle_map(map_size=100):
    grid_map = np.zeros((map_size, map_size), dtype=int)

    # 左上角矩形
    grid_map[10:30, 10:30] = 1

    # 右上角圆形
    center = (75, 25)
    radius = 10
    for y in range(map_size):
        for x in range(map_size):
            if (x - center[0])**2 + (y - center[1])**2 <= radius**2:
                grid_map[y][x] = 1

    # 中间十字
    grid_map[45:55, 40:60] = 1
    grid_map[40:60, 45:55] = 1

    # 左下角小矩形
    grid_map[70:80, 10:20] = 1

    return grid_map

def read_task_points_from_txt(task_file, grid_map):
    task_points = []
    with open(task_file, "r") as f:
        for line in f:
            if line.strip() == "":
                continue
            parts = line.strip().split()
            if len(parts) >= 3:
                x = int(parts[1])
                y = int(parts[2])
                if grid_map[y][x] == 1:
                    print(f"Warning: task point ({x}, {y}) is in obstacle!")
                task_points.append((x, y))
    return task_points

def save_map_and_plot(grid_map, task_points, map_name="map1_structured"):
    os.makedirs("maps", exist_ok=True)

    # 保存地图
    np.save(f"maps/{map_name}.npy", grid_map)

    # 绘图并保存图像
    plt.figure(figsize=(8, 8))
    plt.imshow(grid_map, cmap="Greys", origin="lower")
    xs, ys = zip(*task_points)
    plt.scatter(xs, ys, c='blue', label="Tasks")
    plt.scatter(xs[0], ys[0], c='red', label="Depot (1)")
    plt.title("Structured Obstacle Map with Loaded Tasks")
    plt.legend()
    plt.grid(False)
    plt.savefig(f"maps/{map_name}_plot.png", dpi=300)
    plt.show()

# ========= 主函数示例调用 =========
if __name__ == "__main__":
    map_size = 100
    task_file = "data/point.txt"
    map_name = "map1_structured"

    grid = create_structured_obstacle_map(map_size)
    tasks = read_task_points_from_txt(task_file, grid)
    save_map_and_plot(grid, tasks, map_name=map_name)
