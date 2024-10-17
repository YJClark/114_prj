import itertools
from collections import defaultdict

# 定義狀態的數據結構
class State:
    def __init__(self, target_position, escort_positions):
        self.target_position = target_position
        self.escort_positions = tuple(sorted(escort_positions))  # 將escort位置進行排序，便於處理

    def __hash__(self):
        return hash((self.target_position, self.escort_positions))

    def __eq__(self, other):
        return (self.target_position == other.target_position and
                self.escort_positions == other.escort_positions)

    def __str__(self):
        return f"Target: {self.target_position}, Escorts: {self.escort_positions}"

# 定義動作（上、下、左、右、保持不動）
MOVES = [(0, 1), (0, -1), (1, 0), (-1, 0), (0, 0)]

# 計算新位置
def move(pos, action):
    return (pos[0] + action[0], pos[1] + action[1])

# 判斷位置是否有效
def is_valid(pos, grid_size):
    return 0 <= pos[0] < grid_size and 0 <= pos[1] < grid_size

# 構建動態規劃表格
def build_dp_table(grid_size, target_end_position, max_steps=1000):
    dp_table = defaultdict(lambda: float('inf'))
    dp_actions = {}
    dp_table[State(target_end_position, ())] = 0

    queue = [State(target_end_position, ())]
    steps = 0

    while queue and steps < max_steps:
        steps += 1
        current_state = queue.pop(0)
        current_cost = dp_table[current_state]

        for num_escorts in range(3):  # 考慮 0 到 2 個護衛
            for escort_positions in itertools.combinations(
                [(x, y) for x in range(grid_size) for y in range(grid_size)],
                num_escorts
            ):
                if current_state.target_position in escort_positions:
                    continue

                new_state = State(current_state.target_position, escort_positions)
                
                for actions in itertools.product(MOVES, repeat=num_escorts + 1):
                    new_target_position = move(new_state.target_position, actions[0])
                    if not is_valid(new_target_position, grid_size):
                        continue

                    new_escort_positions = []
                    conflict = False
                    for idx, escort_pos in enumerate(new_state.escort_positions):
                        new_escort_pos = move(escort_pos, actions[idx + 1])
                        if not is_valid(new_escort_pos, grid_size) or new_escort_pos == new_target_position:
                            conflict = True
                            break
                        new_escort_positions.append(new_escort_pos)

                    if conflict:
                        continue

                    next_state = State(new_target_position, new_escort_positions)
                    new_cost = current_cost + 1

                    if new_cost < dp_table[next_state]:
                        dp_table[next_state] = new_cost
                        dp_actions[next_state] = actions
                        queue.append(next_state)

    return dp_table, dp_actions

# 運行動態規劃表格來找最佳路徑
def find_best_path(grid_size, target_start_position, escort_start_positions, dp_table, dp_actions):
    current_state = State(target_start_position, escort_start_positions)
    path = []

    while dp_table[current_state] != 0:  # 終點代價為0，表示已到達終點
        if current_state not in dp_actions:
            raise ValueError(f"State {current_state} not found in dp_actions")
        
        action = dp_actions[current_state]
        path.append(action)

        # 更新狀態
        new_target_position = move(current_state.target_position, action[0])
        new_escort_positions = [move(pos, a) for pos, a in zip(current_state.escort_positions, action[1:])]
        current_state = State(new_target_position, new_escort_positions)

    return path

# 測試演算法
if __name__ == "__main__":
    grid_size = 5
    target_start_position = (4, 4)
    escort_start_positions = [(0, 0), (0, 1)]
    target_end_position = (0, 0)

    # 構建DP表格
    dp_table, dp_actions = build_dp_table(grid_size, target_end_position, max_steps=10000)

    # 找出最佳路徑
    best_path = find_best_path(grid_size, target_start_position, escort_start_positions, dp_table, dp_actions)

    # 打印最佳路徑
    print("最佳路徑動作序列：")
    for step, action in enumerate(best_path):
        print(f"步驟 {step + 1}: {action}")
