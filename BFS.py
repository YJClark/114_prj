from collections import deque

def readFile():
    pass
    # return map_start, map_end

# Function to print the grid
def print_grid(state):
    for row in state:
        print(' '.join(str(x) if x != 0 else '_' for x in row))
    print()


#以下在做post process
def find_movements(prev_step, current_step):
    """
    找出從 prev_step 到 current_step 的所有移動。
    返回一個字典，鍵為方格編號，值為一個元組 ((起始行, 起始列), (目標行, 目標列))。
    """
    movements = {}
    positions_prev = {}
    positions_current = {}

    # 建立方格在 prev_step 和 current_step 中的位置映射
    for i in range(len(prev_step)):
        for j in range(len(prev_step[0])):
            val = prev_step[i][j]
            if val != 0:  # 排除空格
                positions_prev[val] = (i, j)

    for i in range(len(current_step)):
        for j in range(len(current_step[0])):
            val = current_step[i][j]
            if val != 0:  # 排除空格
                positions_current[val] = (i, j)

    # 比較每個方格的位置，找出移動的方格
    for val in positions_prev:
        if val in positions_current:
            if positions_prev[val] != positions_current[val]:
                movements[val] = (positions_prev[val], positions_current[val])
    return movements

def get_path(start, end):
    """
    計算方格從起點到終點的移動路徑。
    只允許水平或垂直移動，不允許對角線移動。
    """
    path = []
    i1, j1 = start
    i2, j2 = end

    # 水平移動
    if i1 == i2:
        step = 1 if j2 > j1 else -1
        for j in range(j1 + step, j2 + step, step):
            path.append((i1, j))
    # 垂直移動
    elif j1 == j2:
        step = 1 if i2 > i1 else -1
        for i in range(i1 + step, i2 + step, step):
            path.append((i, j1))
    else:
        # 對角線移動不允許
        path = []
    return path

def can_merge(current_merged, new_movements, current_positions):
    """
    判斷是否可以將 new_movements 合併到 current_merged
    
    條件：
    1. 每個方格在同一時間只能移動一格
    2. 不允許垂直方向的進出衝突，但允許同方向的同時進出
    3. 移動目標必須是空白格，除非目標格也在同方向移動
    """
    combined = current_merged.copy()
    
    # 檢查條件1：每個方格只能移動一次
    for val, move in new_movements.items():
        if val in combined:
            return False
        combined[val] = move
    
    # 檢查所有移動
    vertical_moves = {}  # 記錄每列的垂直移動 {列: [方向]}
    horizontal_moves = {}  # 記錄每行的水平移動 {行: [方向]}
    
    for val, ((i1, j1), (i2, j2)) in combined.items():
        # 確保每次只移動一格
        if abs(i2 - i1) + abs(j2 - j1) > 1:
            return False
        
        # 記錄移動方向
        if i1 != i2:  # 垂直移動
            direction = 1 if i2 > i1 else -1
            if j1 not in vertical_moves:
                vertical_moves[j1] = []
            vertical_moves[j1].append(direction)
        elif j1 != j2:  # 水平移動
            direction = 1 if j2 > j1 else -1
            if i1 not in horizontal_moves:
                horizontal_moves[i1] = []
            horizontal_moves[i1].append(direction)
    
    # 檢查條件2：垂直方向的進出衝突
    for column, directions in vertical_moves.items():
        if len(set(directions)) > 1:  # 如果同一列有不同方向的移動
            return False
    
    # 檢查條件3：移動目標必須是空白格或同向移動的方格
    for val, ((i1, j1), (i2, j2)) in combined.items():
        target_pos = (i2, j2)
        if target_pos in current_positions:
            target_val = current_positions[target_pos]
            if target_val not in combined:
                return False  # 目標位置有靜止的方格
            else:
                # 檢查目標方格是否同向移動
                target_move = combined[target_val]
                target_dir = (
                    target_move[1][0] - target_move[0][0],
                    target_move[1][1] - target_move[0][1]
                )
                current_dir = (i2 - i1, j2 - j1)
                if target_dir != current_dir:
                    return False  # 目標方格移動方向不同
    
    return True


def merge_steps(steps):
    """
    將多個步驟合併為最少的步驟數
    """
    all_movements = []
    for k in range(1, len(steps)):
        movements = find_movements(steps[k-1], steps[k])
        all_movements.append(movements)
    
    merged_steps = []
    merged_steps_info = []
    current_merged = {}
    current_positions = {}
    
    # 初始化當前方格位置
    for i in range(len(steps[0])):
        for j in range(len(steps[0][0])):
            val = steps[0][i][j]
            if val != 0:
                current_positions[(i, j)] = val
    
    current_step_indices = []
    
    for step_idx, movements in enumerate(all_movements):
        if not movements:
            continue
        
        if can_merge(current_merged, movements, current_positions):
            current_step_indices.append(step_idx + 1)
            # 更新當前合併的移動和位置
            for val, ((i1, j1), (i2, j2)) in movements.items():
                current_merged[val] = ((i1, j1), (i2, j2))
                del current_positions[(i1, j1)]
                current_positions[(i2, j2)] = val
        else:
            if current_merged:
                merged_steps.append(current_merged)
                merged_steps_info.append(current_step_indices)
            current_merged = movements
            current_step_indices = [step_idx + 1]
            # 重置並更新位置
            current_positions = {}
            for i in range(len(steps[step_idx])):
                for j in range(len(steps[step_idx][0])):
                    val = steps[step_idx][i][j]
                    if val != 0:
                        current_positions[(i, j)] = val
    
    if current_merged:
        merged_steps.append(current_merged)
        merged_steps_info.append(current_step_indices)
    
    return merged_steps, merged_steps_info

def print_merged_steps(merged_steps, merged_steps_info):
    for idx, (step, info) in enumerate(zip(merged_steps, merged_steps_info), 1):
        steps_str = ', '.join([f"{s} step" for s in info])
        print(f"Steps after merge step {idx} (includes {steps_str}):")
        for val, ((i1, j1), (i2, j2)) in step.items():
            print(f"   {val} from ({i1}, {j1}) to ({i2}, {j2})")
        print()

def print_final_grids(steps, merged_steps, merged_steps_info):
    print("initial state:")
    print_grid(steps[0])

    current_grid = [row.copy() for row in steps[0]]
    for idx, (step, info) in enumerate(zip(merged_steps, merged_steps_info), 1):
        for val, ((i1, j1), (i2, j2)) in step.items():
            current_grid[i1][j1] = 0
            current_grid[i2][j2] = val
        print(f"Merge step {idx}:")
        print_grid(current_grid)



#以下在做BFS
def bfs_min_moves(start, target):
    size = (len(start), len(start[0]))
    # Convert the puzzle to tuples for hashable states
    start_state = tuple(tuple(row) for row in start)
    target_state = tuple(tuple(row) for row in target)
    
    # Directions for moving (up, down, left, right)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # Set to keep track of visited states
    visited = set()
    visited.add(start_state)
    
    # Queue for BFS, storing (current state, distance, and the parent state)
    queue = deque([(start_state, 0, None)])  # (state, distance, parent)
    
    # Dictionary to record parent states
    parent_map = {start_state: None}   # {current_state: previous_state}
    
    while queue:
        current_state, depth, parent = queue.popleft()

        # Check if the current state matches the target state
        if all(
            current_state[i][j] == target_state[i][j]
            for i in range(size[0]) 
            for j in range(size[1])
            if target_state[i][j] > 0  # Only consider target positions with > 0
        ):
            # Reconstruct the path from start to goal
            path = []
            while current_state is not None:
                path.append(current_state)
                current_state = parent_map[current_state]        

            # Convert each state from tuple of tuples to list of lists
            path_re = [ [list(row) for row in state] for state in reversed(path) ]

            print("Steps before merging:")
            # Print each step from start to goal
            for state in path_re:    
                print_grid(state)

            # Perform post-processing
            merged_steps, merged_steps_info = merge_steps(path_re)

            print("Which steps can be merged:")
            print_merged_steps(merged_steps, merged_steps_info)

            print("Changes:")
            print_final_grids(path_re, merged_steps, merged_steps_info)

            return depth
        
        # Find positions of zeroes (available slots)
        empty_positions = [(i, j) for i in range(size[0]) for j in range(size[1]) if current_state[i][j] == 0]
        
        # Generate all possible moves by shifting non-zero blocks into empty slots
        for empty_pos in empty_positions:
            x, y = empty_pos
            for direction in directions:
                new_x, new_y = x + direction[0], y + direction[1]
                if 0 <= new_x < size[0] and 0 <= new_y < size[1] and current_state[new_x][new_y] != 0:
                    # Make a move (swap non-zero with zero)
                    new_state = [list(row) for row in current_state]  # Convert tuple to list
                    new_state[x][y], new_state[new_x][new_y] = new_state[new_x][new_y], new_state[x][y]
                    new_state_tuple = tuple(tuple(row) for row in new_state)
                    
                    # If this state hasn't been visited, enqueue it
                    if new_state_tuple not in visited:
                        visited.add(new_state_tuple)
                        parent_map[new_state_tuple] = current_state  # Record parent state
                        queue.append((new_state_tuple, depth + 1, current_state))
    return -1  # Return -1 if no solution found


def main():

# id8
    # map_start = [
    #     [1, 5, 3],
    #     [-1, 0, 2],
    #     [-1, 4, 0]
    # ]

    # map_end = [
    #     [ 4, 1, 0],
    #     [-1, 0, 5],
    #     [-1, 3, 2]
    # ]
    # map_start = [
    #     [3, 5, 2],
    #     [0, 1, -1],
    #     [4, 0, -1]
    # ]

    # map_end = [
    #     [2, 1, 5],
    #     [3, 4, -1],
    #     [0, 0, -1]
    # ]

#無解測資 tar8 sp1 id5
    # map_start = [
    #     [8, 5, 0],
    #     [4, 6, 3],
    #     [2, 1, 7]
    # ]

    # map_end = [
    #     [7, 2, 0],
    #     [3, 8, 4],
    #     [1, 5, 6]
    # ]

    # tar8 sp2 id 1
    map_start = [
        [-1, 8, 4, 0],
        [1,-1,-1,-1],
        [5, 7, 2, 0],
        [6, 3,-1,-1]
    ]
    map_end = [
        [-1, 0, 6, 2],
        [7,-1,-1,-1],
        [8, 5, 1, 3],
        [4, 0,-1,-1]
    ]

# Run BFS to find the minimum number of moves and print each step
# map_start, map_end = readFile()
    min_moves = bfs_min_moves(map_start, map_end)
    print(f"Total moves: {min_moves}")  

if __name__ == "__main__":
    main()