from collections import deque

# Function to print the 3x3 grid
def print_grid(state):
    for row in state:
        print(' '.join(str(x) if x != 0 else '_' for x in row))
    print()

# def print_queue(queue):
    # print('deque(')
    # for i in queue:
    #     print(f'\t{i}')
    # print(')')

# BFS function to find the shortest number of moves and print each step
def bfs_min_moves(start, target):
    size = (len(start), len(start[0]))
    # Convert the puzzle to tuples for hashable states
    start_state = tuple(tuple(row) for row in start)   # ((2, 3, 1), (4, 5, -1), (0, -1, 0))
    target_state = tuple(tuple(row) for row in target)
    # for t in target:
    #     target_state.append(tuple(tuple(row) for row in t)) # ((0, -1, 0), (5, -1, 3), (2, 4, 1))
    
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

        # all() 用來檢查一個可迭代對象中的所有元素是否都為 True。
        # if current_state == target_state:
        if all(
            current_state[i][j] == target_state[i][j] for i in range(size[0]) for j in range(size[1]) if target_state[i][j] > 0  # 只考慮目標狀態中大於 0 的數字
        ):
            # If we've reached the goal, reconstruct the path
            path = []
            while current_state is not None:
                path.append(current_state)
                current_state = parent_map[current_state]
            
            # Print each step from start to goal
            for state in reversed(path):
                print_grid(state)
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
                    new_state = [list(row) for row in current_state]  # Copy current state
                    new_state[x][y], new_state[new_x][new_y] = new_state[new_x][new_y], new_state[x][y]
                    new_state_tuple = tuple(tuple(row) for row in new_state)
                    
                    # If this state hasn't been visited, enqueue it
                    if new_state_tuple not in visited:
                        visited.add(new_state_tuple)
                        parent_map[new_state_tuple] = current_state  # Record parent state
                        queue.append((new_state_tuple, depth + 1, current_state))
        # print(queue[0])
    
    return -1  # Return -1 if no solution found

# Example puzzle（比 model_obs 快很多）:
# map_start = [
#     [2, 3, 1],
#     [4, 5, -1],
#     [0, -1, 0]
# ]
# map_end = [
#     [0, -1, 0],
#     [5, -1, 3],
#     [2, 4, 1]
# ]

map_start = [
	[0, 1, 0],
	[-1, 2, -1],
	[3, 4, 0]
]
map_end = [
	[2, 4, 0],
	[-1, 0, -1],
	[1, 0, 3]
]
# Find and print all unique valid end states
# valid_end_states = find_all_valid_end_states(map_end)
# print(f"Number of unique valid end states: {len(valid_end_states)}")

# map_end = valid_end_states

# Run BFS to find the minimum number of moves and print each step
min_moves = bfs_min_moves(map_start, map_end)
print(f"Minimum moves: {min_moves}")