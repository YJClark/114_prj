# 目前一次只動一台
# 還沒要移動或完成移動的都視為障礙物
# 求出完整做完的最短路徑(cost)
import random
import math
from collections import deque

initial_positions = [(1, 1), (3, 1), (3, 3), (4, 2), (2, 4)]
goal_positions = [(5, 5), (4, 5), (3, 5), (2, 5), (1, 5)]
 
directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上, 下, 左, 右


# in grid and not obstacles
def is_valid_position(position, obstacles, grid_size=(5, 5)):
    return (1 <= position[0] <= grid_size[0] and 1 <= position[1] <= grid_size[1] 
            and position not in obstacles)

# 移動車車
def move_car(position, direction):
    return (position[0] + direction[0], position[1] + direction[1])

# BFS、考慮障礙物
def bfs_shortest_path(start, goal, obstacles):
    queue = deque([(start, [])])
    visited = set()
    visited.add(start)
    
    while queue:
        current_position, path = queue.popleft()
        
        if current_position == goal:
            return path + [current_position]
        
        for direction in directions:
            next_position = move_car(current_position, direction)
            if is_valid_position(next_position, obstacles) and next_position not in visited:
                visited.add(next_position)
                queue.append((next_position, path + [current_position]))
    return []

# 計算total cost
def calculate_total_cost(paths):
    return sum(len(path) for path in paths)

# 生成鄰域解：隨機交換兩輛車的移動順序
def generate_neighbor_solution(paths, order):
    new_order = order[:]
    i, j = random.sample(range(len(order)), 2)
    new_order[i], new_order[j] = new_order[j], new_order[i]
    
    new_paths = []
    obstacles = set(initial_positions)
    
    for idx in new_order:
        start = initial_positions[idx]
        goal = goal_positions[idx]
        obstacles.remove(start)  # 移動車輛不再是障礙物
        path = bfs_shortest_path(start, goal, obstacles)
        new_paths.append(path)
        obstacles.add(goal)  # 移動到goal position後，該位置成為新的障礙物
    
    return new_paths, new_order

# SA主函數
def simulated_annealing(initial_temp, cooling_rate):
    current_order = list(range(len(initial_positions)))
    current_paths = []
    obstacles = set(initial_positions)
    
    for idx in current_order:
        start = initial_positions[idx]
        goal = goal_positions[idx]
        obstacles.remove(start)  # 移動車輛不再是障礙物
        path = bfs_shortest_path(start, goal, obstacles)
        current_paths.append(path)
        obstacles.add(goal)  # 移動到目標位置後，該位置成為新的障礙物
    
    current_cost = calculate_total_cost(current_paths)
    best_paths = current_paths[:]
    best_cost = current_cost
    best_order = current_order[:]
    temperature = initial_temp
    
    while temperature > 1:
        new_paths, new_order = generate_neighbor_solution(current_paths, current_order)
        new_cost = calculate_total_cost(new_paths)
        
        # 若新解更好，或以一定機率接受更差的解
        if new_cost < current_cost or random.uniform(0, 1) < math.exp((current_cost - new_cost) / temperature):
            current_paths = new_paths
            current_order = new_order
            current_cost = new_cost
        
        # 更新最佳解
        if current_cost < best_cost:
            best_paths = current_paths[:]
            best_order = current_order[:]
            best_cost = current_cost
        
        # 降溫
        temperature *= cooling_rate
    
    return best_paths, best_order, best_cost

# hyper parameters
initial_temp = 500
cooling_rate = 0.98

best_paths, best_order, best_cost = simulated_annealing(initial_temp, cooling_rate)

# result
print("Total cost(distance): ", best_cost)
print("Best moving order:", [x + 1 for x in best_order])
for i, path in enumerate(best_paths):
    print(f"Order {i + 1}'s path: {path}")










# import numpy as np
# import random

# 定義車輛位置及目標位置
# initial_positions = [(1, 1), (3, 1), (3, 3)]
# goal_positions = [(5, 5), (5, 4), (5, 3)]
# obstacle_positions = [(2, 1), (3, 2), (4, 4)]  # 障礙物初始位置


# # 計算曼哈頓距離
# def manhattan_distance(pos1, pos2):
#     return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

# # 生成初始解
# def generate_initial_solution():
#     order = list(range(len(initial_positions)))
#     random.shuffle(order)
#     return order

# # 計算路徑的總成本並記錄每個車輛的移動路徑
# def total_cost(order):
#     total_distance = 0
#     paths = []  # 用於記錄每輛車的移動路徑

#     for i in range(len(order)):
#         car_position = initial_positions[order[i]]
#         goal_position = goal_positions[i]
#         path = [car_position]
#         total_distance += manhattan_distance(car_position, goal_position)

#         # 模擬車輛直線移動，記錄每一步的位置
#         while car_position != goal_position:
#             if car_position[0] < goal_position[0]:
#                 car_position = (car_position[0] + 1, car_position[1])
#             elif car_position[0] > goal_position[0]:
#                 car_position = (car_position[0] - 1, car_position[1])
#             elif car_position[1] < goal_position[1]:
#                 car_position = (car_position[0], car_position[1] + 1)
#             elif car_position[1] > goal_position[1]:
#                 car_position = (car_position[0], car_position[1] - 1)
#             path.append(car_position)
        
#         paths.append(path)

#         # 加入障礙物的移動成本
#         for obstacle in obstacle_positions:
#             total_distance += manhattan_distance(car_position, obstacle)
    
#     return total_distance, paths

# # 產生鄰居解，考慮障礙物移動
# def generate_neighbor(solution):
#     neighbor = solution[:]
#     i, j = random.sample(range(len(solution)), 2)
#     neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
    
#     # 隨機選擇是否移動障礙物
#     if random.random() > 0.5:
#         obstacle_index = random.choice(range(len(obstacle_positions)))
#         new_position = (random.choice(range(1, 5)), random.choice(range(1, 5)))
#         obstacle_positions[obstacle_index] = new_position
    
#     return neighbor

# # 模擬退火演算法
# def simulated_annealing(initial_temp, cooling_rate, max_iterations):
#     current_solution = generate_initial_solution()
#     current_cost, current_paths = total_cost(current_solution)
#     temp = initial_temp
    
#     best_solution = current_solution[:]
#     best_cost = current_cost
#     best_paths = current_paths[:]
    
#     for i in range(max_iterations):
#         neighbor = generate_neighbor(current_solution)
#         neighbor_cost, neighbor_paths = total_cost(neighbor)
        
#         if neighbor_cost < current_cost or random.random() < np.exp((current_cost - neighbor_cost) / temp):
#             current_solution = neighbor
#             current_cost = neighbor_cost
#             current_paths = neighbor_paths
            
#             if neighbor_cost < best_cost:
#                 best_solution = neighbor
#                 best_cost = neighbor_cost
#                 best_paths = neighbor_paths
        
#         temp *= cooling_rate
    
#     return best_solution, best_cost, best_paths

# # 設定參數並運行退火演算法
# initial_temp = 1000
# cooling_rate = 0.99
# max_iterations = 10000

# best_solution, best_cost, best_paths = simulated_annealing(initial_temp, cooling_rate, max_iterations)

# # result
# print("best moving order: ", best_solution)
# print("total cost(distance): ", best_cost)
# print("Each car's path: ")
# for i, path in enumerate(best_paths):
#     print(f"car {best_solution[i]+1} path: {path}")



# 2024 sa pseudocode
# import math
# import random

# def SP(S):
#     # 這裡需要根據具體問題定義目標函數
#     # 例如，可以返回S中元素的和作為示例
#     return sum(S)

# def Inversion(S):
#     # 這裡需要根據具體問題定義如何生成新解
#     # 例如，可以隨機交換兩個元素的位置
#     Snew = S.copy()
#     i, j = random.sample(range(len(S)), 2)
#     Snew[i], Snew[j] = Snew[j], Snew[i]
#     return Snew

# def SA(S, T, tmin, K, n_iteration):
#     Obj = SP(S)
#     best_S, best_Obj = S, Obj

#     while T >= tmin:
#         for _ in range(n_iteration):
#             Snew = Inversion(S)
#             Obj_new = SP(Snew)
#             delta = Obj_new - Obj

#             if delta < 0:
#                 S, Obj = Snew, Obj_new
#                 if Obj_new < best_Obj:
#                     best_S, best_Obj = Snew, Obj_new
#             else:
#                 R = random.random()
#                 E = math.exp(-delta / T)
#                 if R < E:
#                     S, Obj = Snew, Obj_new

#         T = K * T

#     return best_S, best_Obj

# # 測試代碼
# if __name__ == "__main__":
#     # 初始解（這裡用一個列表作為例子）
#     initial_solution = [1, 2, 3, 4, 5]
    
#     # 參數設置
#     initial_temperature = 100
#     min_temperature = 0.1
#     cooling_rate = 0.95
#     iterations_per_temperature = 100

#     best_solution, best_objective = SA(initial_solution, initial_temperature, min_temperature, cooling_rate, iterations_per_temperature)

#     print("Best solution:", best_solution)
#     print("Best objective value:", best_objective)