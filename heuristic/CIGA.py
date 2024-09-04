# reference: An improved genetic algorithm with co-evolutionary strategy for global path planning of multiple mobile robots
# https://shorturl.at/0aLWh
#還沒有加上菁英個體的保留

import numpy as np
import random

# GA 和 CIGA 參數設置
POPULATION_SIZE = 80
MAX_GENERATIONS = 200
CROSSOVER_PROBABILITY = 0.8
MUTATION_PROBABILITY = 0.15
NUM_ROBOTS = 3  # 假設有三台機器人

# 創建環境，使用10x10的網格
def create_environment():
    environment = np.zeros((10, 10))
    obstacles = [(3, 3), (3, 4), (4, 4), (5, 5), (6, 6)]
    for obs in obstacles:
        environment[obs] = 1
    return environment

# 編碼和解碼路徑
def encode_path(path):
    return [p[0] * 10 + p[1] for p in path]

def decode_path(encoded_path):
    return [(int(p / 10), p % 10) for p in encoded_path]

# 計算適應度函數
def fitness_function(path, environment):
    path_length = sum(np.sqrt((path[i][0] - path[i-1][0])**2 + (path[i][1] - path[i-1][1])**2) for i in range(1, len(path)))
    safety = sum(np.min([np.linalg.norm(np.array(p) - np.array(obs)) for obs in np.argwhere(environment == 1)]) for p in path)
    smoothness = sum(np.abs(path[i][0] - 2 * path[i-1][0] + path[i-2][0]) + np.abs(path[i][1] - 2 * path[i-1][1] + path[i-2][1]) for i in range(2, len(path)))
    return 1 / (path_length + safety + smoothness)

# 選擇操作
def selection(population, fitnesses):
    selected_index = random.choices(range(len(population)), weights=fitnesses, k=1)[0]
    return population[selected_index]

# 交叉操作
def crossover(parent1, parent2):
    if random.random() < CROSSOVER_PROBABILITY:
        cross_point = random.randint(1, min(len(parent1), len(parent2)) - 2)
        child1 = parent1[:cross_point] + parent2[cross_point:]
        child2 = parent2[:cross_point] + parent1[cross_point:]
        return child1, child2
    return parent1, parent2

# 突變操作
def mutation(path):
    if random.random() < MUTATION_PROBABILITY:
        mutate_point = random.randint(0, len(path) - 1)
        path[mutate_point] = random.choice(range(10*10))
    return path

# 交叉干擾函數，檢查路徑是否衝突
# def cross_interference(path1, path2):
#     for i in range(min(len(path1), len(path2))):
#         if path1[i] == path2[i]:  # 簡單檢查是否在同一時間點到達同一位置
#             return True
#     return False

# 判斷路徑是否相交並設置衝突係數
def cross_interference(path1, path2):
    intersecting = False
    conflicting = False
    for i in range(min(len(path1), len(path2))):
        if path1[i] == path2[i]:
            intersecting = True
            if i == len(path1) - 1 or i == len(path2) - 1:  # 同時到達終點不算衝突
                continue
            conflicting = True
            break
    if conflicting:
        return 10  # 衝突
    elif intersecting:
        return 0  # 相交但不衝突
    return 0  # 不相交也不衝突

# CIGA主函數
def CIGA(environment, start_positions, end_positions):
    path_lengths = [random.randint(5, 15) for _ in range(NUM_ROBOTS)]
    populations = [
        [
            encode_path(
                [start_positions[robot_index]] +
                [(random.randint(0, 9), random.randint(0, 9)) for _ in range(path_lengths[robot_index] - 2)] +
                [end_positions[robot_index]]
            ) for _ in range(POPULATION_SIZE)
        ]
        for robot_index in range(NUM_ROBOTS)
    ]
    
    for generation in range(MAX_GENERATIONS):
        new_populations = []
        for robot_index in range(NUM_ROBOTS):
            fitnesses = [fitness_function(decode_path(individual), environment) for individual in populations[robot_index]]
            new_population = []
            for _ in range(POPULATION_SIZE // 2):
                parent1 = selection(populations[robot_index], fitnesses)
                parent2 = selection(populations[robot_index], fitnesses)
                child1, child2 = crossover(parent1, parent2)
                child1 = mutation(child1)
                child2 = mutation(child2)
                new_population.extend([child1, child2])
            new_populations.append(new_population)

        # 協同演化部分
        for i in range(NUM_ROBOTS):
            for j in range(i + 1, NUM_ROBOTS):
                for ind_i in new_populations[i]:
                    for ind_j in new_populations[j]:
                        # 使用cross_interference函數的返回值來決定如何處理衝突
                        interference_value = cross_interference(ind_i, ind_j)
                        if interference_value > 0:  # 表示有衝突
                            # 簡單解決衝突：重新隨機初始化一個解
                            new_populations[j][new_populations[j].index(ind_j)] = encode_path(
                                [start_positions[j]] +
                                [(random.randint(0, 9), random.randint(0, 9)) for _ in range(path_lengths[j] - 2)] +
                                [end_positions[j]]
                            )

        populations = new_populations
        
        #這邊應該可以刪掉
        # 只打印第1個機器人的最佳適應度值
        #best_fitness = max([fitness_function(decode_path(ind), environment) for ind in populations[0]])
        #print(f"Generation {generation}: Best Fitness = {best_fitness}")

    # 印出每個機器人最終的最佳路徑
    for robot_index in range(NUM_ROBOTS):
        best_individual = max(populations[robot_index], key=lambda ind: fitness_function(decode_path(ind), environment))
        best_path = decode_path(best_individual)
        print(f"Robot {robot_index + 1} final path: {best_path}")

if __name__ == "__main__":
    env = create_environment()
    
    # 指定每個機器人的起點和終點
    start_positions = [(0, 0), (9, 9), (0, 9)]
    end_positions = [(9, 0), (0, 9), (9, 9)]
    
    CIGA(env, start_positions, end_positions)
