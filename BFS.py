from collections import deque

def readFile():
    pass
    # return map_start, map_end

# Function to print the 3x3 grid
def print_grid(state):
    for row in state:
        print(' '.join(str(x) if x != 0 else '_' for x in row))
    print()

def print_queue(queue):
    print('deque(')
    for i in queue:
        print(f'\t{i}')
    print(')')

# postprocess: merge the moves
def merge_moves(path):
    path = list(reversed(path))     #gotta reverse the list
    merged_path = []
    merged_path.append(path[0])     #store the initial state first
 
    # for k in range(len(path)):
    #     for i in range(len(path[k])):
    #         for j in range(len(path[k])):
    #             if path[k][i][j] != path[k+1][i][j]:
    #                 # Calculate the difference in positions
    #                 diff_i = i - path[k+1].index(path[k][i][j])
    #                 diff_j = j - path[k+1][j].index(path[k][i][j])
    #                 # Check if the difference is within 1 grid
    #                 if abs(diff_i) + abs(diff_j) <= 1:
    #                     # If within 1 grid, it's a valid merge
    #                     merged_path.append(path[k+1][i][j])
    #                 else:
    #                     # If not within 1 grid, it's not a valid merge
    #                     pass
    # if 

    #     can_merge(path[i], path[i+1])
    pass

def can_merge(path1, path2):
    pass

# def merge_moves(moves):
#     merged = []
#     current_move = None

#     for move in moves:
#         if not current_move:
#             current_move = move
#         elif can_merge(current_move, move):
#             current_move = merge_single_move(current_move, move)
#         else:
#             merged.append(current_move)
#             current_move = move

#     if current_move:
#         merged.append(current_move)

#     return merged

# def can_merge(move1, move2):
#     # 检查两个移动是否可以合并
#     return all(
#         abs(m1) + abs(m2) <= 1 and not (m1 * m2 < 0)
#         for m1, m2 in zip(move1, move2)
#     )

# def merge_single_move(move1, move2):
#     # 合并两个移动
#     return tuple(m1 + m2 for m1, m2 in zip(move1, move2))



# BFS function to find the shortest number of moves and print each step
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
            
            #do post_process here
            merge_moves(path)

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

# 測資改讀檔
# Example puzzle（比 model_obs 快很多）:
#5_2.3
map_start = [
	[3, 5, 2],
	[0, 1, -1],
	[4, 0, -1]
]

map_end = [
	[2, 1, 5],
	[3, 4, -1],
	[0, 0, -1]
]

# map_start = [
#     [2, 3, -1],
#     [-1,-1, 0],
#     [4, -1, 1]
# ]
# map_end = [
#     [4, 1, -1],
#     [-1,-1, 2],
#     [3, -1, 0]
# ]

# 4*4 tar8 sp2 id 1
# map_start = [
#     [-1, 8, 4, 0],
#     [1,-1,-1,-1],
#     [5, 7, 2, 0],
#     [6, 3,-1,-1]
# ]
# map_end = [
#     [-1, 0, 6, 2],
#     [7,-1,-1,-1],
#     [8, 5, 1, 3],
#     [4, 0,-1,-1]
# ]

#graph_start_5x5.txt

# map_start = [
#     [1, 3,-1,-1,-1],
#     [2, 4,-1,-1,-1],
#     [5,-1,-1,-1,-1],
#     [-1,-1,-1,-1,-1],
#     [-1,-1,-1,-1, 0]

# ]
# map_end = [
#     [4, 0,-1,-1,-1],
#     [3, 5,-1,-1,-1],
#     [2,-1,-1,-1,-1],
#     [-1,-1,-1,-1,-1],
#     [-1,-1,-1,-1, 1]
# ]


# Run BFS to find the minimum number of moves and print each step
# map_start, map_end = readFile()
min_moves = bfs_min_moves(map_start, map_end)
print(f"Total moves: {min_moves}")  