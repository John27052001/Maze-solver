import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# Define the maze structure (1 = wall, 0 = path)
maze = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 0, 1, 1, 0, 1],
    [1, 0, 1, 0, 0, 0, 0, 1, 0, 1],
    [1, 0, 1, 1, 1, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
    [1, 0, 0, 1, 0, 0, 0, 1, 0, 1],
    [1, 0, 1, 1, 1, 1, 0, 1, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
])

start = (1, 1)  # Start position
end = (8, 8)    # End position

def dfs(maze, start, end):
    stack = [(start, [start])]  # Stack stores (current position, path taken)
    visited = set()

    while stack:
        (x, y), path = stack.pop()
        if (x, y) == end:
            return path

        if (x, y) in visited:
            continue
        visited.add((x, y))

        # Explore neighbors (up, down, left, right)
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < maze.shape[0] and 0 <= new_y < maze.shape[1] and maze[new_x, new_y] == 0:
                stack.append(((new_x, new_y), path + [(new_x, new_y)]))
    
    return None  # No path found

def bfs(maze, start, end):
    queue = deque([(start, [start])])  # Queue stores (current position, path taken)
    visited = set()

    while queue:
        (x, y), path = queue.popleft()
        if (x, y) == end:
            return path

        if (x, y) in visited:
            continue
        visited.add((x, y))

        # Explore neighbors (up, down, left, right)
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < maze.shape[0] and 0 <= new_y < maze.shape[1] and maze[new_x, new_y] == 0:
                queue.append(((new_x, new_y), path + [(new_x, new_y)]))
    
    return None  # No path found

def plot_maze(maze, dfs_path=None, bfs_path=None):
    plt.figure(figsize=(6, 6))
    plt.imshow(maze, cmap='gray_r')

    # Plot DFS path
    if dfs_path:
        x, y = zip(*dfs_path)
        plt.plot(y, x, color='blue', marker='o', label='DFS Path')

    # Plot BFS path
    if bfs_path:
        x, y = zip(*bfs_path)
        plt.plot(y, x, color='red', marker='o', linestyle='dashed', label='BFS Path')

    # Mark start and end points
    plt.scatter(start[1], start[0], color='green', s=100, label="Start")
    plt.scatter(end[1], end[0], color='purple', s=100, label="End")

    plt.legend()
    plt.title("Maze Solver Visualization")
    plt.show()

# Solve using DFS and BFS
dfs_path = dfs(maze, start, end)
bfs_path = bfs(maze, start, end)

# Plot the results
plot_maze(maze, dfs_path, bfs_path)
