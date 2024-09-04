import numpy as np
import random
import math

# 參數設置
n = 5  # 網格大小
r = 3  # 貨物數量
k = 2  # 機器人數量
T_max = 1000  # 最大時間周期
initial_temperature = 1000  # 初始溫度
alpha = 0.95  # 降溫係數
max_iterations = 1000  # 最大迭代次數

# 初始化位置
cargo_positions = [(0, 0), (4, 4), (2, 2)]
target_positions = [(4, 4), (0, 0), (3, 3)]
robot_positions = [(1, 1), (3, 3)]

# 計算目標函數 (最大完成時間)
def objective_function(solution):
    max_time = 0
    for cargo in solution['cargo_paths']:
        max_time = max(max_time, len(cargo))
    return max_time

# 生成隨機解
def generate_random_solution():
    solution = {
        'robot_paths': [],
        'cargo_paths': []
    }
    
    for i in range(k):
        path = [(random.randint(0, n-1), random.randint(0, n-1)) for _ in range(T_max)]
        solution['robot_paths'].append(path)
    
    for i in range(r):
        path = [(random.randint(0, n-1), random.randint(0, n-1)) for _ in range(T_max)]
        solution['cargo_paths'].append(path)
    
    return solution

# 生成鄰域解 (隨機改變一個貨物或機器人的路徑)
def generate_neighbor_solution(solution):
    neighbor = {
        'robot_paths': [list(path) for path in solution['robot_paths']],
        'cargo_paths': [list(path) for path in solution['cargo_paths']]
    }
    
    if random.random() < 0.5:
        robot_index = random.randint(0, k-1)
        time_index = random.randint(0, T_max-1)
        neighbor['robot_paths'][robot_index][time_index] = (random.randint(0, n-1), random.randint(0, n-1))
    else:
        cargo_index = random.randint(0, r-1)
        time_index = random.randint(0, T_max-1)
        neighbor['cargo_paths'][cargo_index][time_index] = (random.randint(0, n-1), random.randint(0, n-1))
    
    return neighbor

# 退火演算法
def simulated_annealing():
    current_solution = generate_random_solution()
    current_cost = objective_function(current_solution)
    
    temperature = initial_temperature
    best_solution = current_solution
    best_cost = current_cost
    
    for iteration in range(max_iterations):
        neighbor_solution = generate_neighbor_solution(current_solution)
        neighbor_cost = objective_function(neighbor_solution)
        
        if neighbor_cost < current_cost:
            current_solution = neighbor_solution
            current_cost = neighbor_cost
        else:
            acceptance_probability = math.exp(-(neighbor_cost - current_cost) / temperature)
            if random.random() < acceptance_probability:
                current_solution = neighbor_solution
                current_cost = neighbor_cost
        
        if current_cost < best_cost:
            best_solution = current_solution
            best_cost = current_cost
        
        temperature *= alpha  # 降低溫度
    
    return best_solution, best_cost

best_solution, best_cost = simulated_annealing()

#print("objective_value: ", best_cost)
print("robot_path: ", best_solution['robot_paths'])
#print("cargo_path: ", best_solution['cargo_paths'])
