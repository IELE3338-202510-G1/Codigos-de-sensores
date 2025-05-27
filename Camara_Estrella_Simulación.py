import cv2
import numpy as np
import heapq
import math
import time
from collections import deque

class Node:
    def __init__(self, position, parent=None):
        self.position = position
        self.parent = parent
        self.g = self.h = self.f = 0
    def __lt__(self, other):
        return self.f < other.f

def inflate_obstacles(maze, radius):
    inflated = maze.copy()
    for y in range(maze.shape[0]):
        for x in range(maze.shape[1]):
            if maze[y, x] == 1:
                for dy in range(-radius, radius+1):
                    for dx in range(-radius, radius+1):
                        ny, nx = y+dy, x+dx
                        if 0 <= ny < maze.shape[0] and 0 <= nx < maze.shape[1]:
                            inflated[ny, nx] = 1
    return inflated

def astar(maze, start, end):
    open_list = []
    closed = set()
    heapq.heappush(open_list, Node(start))

    while open_list:
        current = heapq.heappop(open_list)
        if current.position == end:
            path = []
            while current:
                path.append(current.position)
                current = current.parent
            return path[::-1]
        closed.add(current.position)

        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            ny, nx = current.position[0] + dy, current.position[1] + dx
            if 0 <= ny < maze.shape[0] and 0 <= nx < maze.shape[1]:
                if maze[ny, nx] == 0 and (ny, nx) not in closed:
                    neighbor = Node((ny, nx), current)
                    neighbor.g = current.g + 1
                    neighbor.h = abs(ny - end[0]) + abs(nx - end[1])
                    neighbor.f = neighbor.g + neighbor.h
                    heapq.heappush(open_list, neighbor)
    return None

def find_closest_free_cell(maze, start):
    visited = set()
    queue = deque([start])
    visited.add(start)
    while queue:
        y, x = queue.popleft()
        if maze[y, x] == 0:
            return (y, x)
        for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
            ny, nx = y+dy, x+dx
            if 0 <= ny < maze.shape[0] and 0 <= nx < maze.shape[1] and (ny, nx) not in visited:
                visited.add((ny, nx))
                queue.append((ny, nx))
    return None

def find_closest_ordered_points(origin, targets):
    remaining = targets.copy()
    path = []
    current = origin
    while remaining:
        nearest = min(remaining, key=lambda p: abs(p[0]-current[0]) + abs(p[1]-current[1]))
        path.append(nearest)
        current = nearest
        remaining.remove(nearest)
    return path

def smooth_path(path, max_angle_change_deg=30):
    def angle_between(p1, p2):
        return math.atan2(p2[0]-p1[0], p2[1]-p1[1])
    
    if len(path) < 3:
        return path

    smoothed = [path[0]]
    for i in range(1, len(path)-1):
        a1 = angle_between(path[i-1], path[i])
        a2 = angle_between(path[i], path[i+1])
        diff = abs(math.degrees(a2 - a1)) % 360
        if diff > 180:
            diff = 360 - diff
        if diff > max_angle_change_deg:
            mid = ((path[i][0] + path[i+1][0]) / 2, (path[i][1] + path[i+1][1]) / 2)
            smoothed.append(path[i])
            smoothed.append(mid)
        else:
            smoothed.append(path[i])
    smoothed.append(path[-1])
    return smoothed

def interpolate_path(path, step=0.2):
    interp = []
    for i in range(len(path)-1):
        p1 = path[i]
        p2 = path[i+1]
        dist = math.dist(p1, p2)
        steps = max(int(dist / step), 1)
        for s in range(steps):
            t = s/steps
            y = (1 - t)*p1[0] + t*p2[0]
            x = (1 - t)*p1[1] + t*p2[1]
            interp.append((y,x))
    interp.append(path[-1])
    return interp

def smooth_interpolated_path(path, window_size=5):
    smoothed = []
    for i in range(len(path)):
        y_vals = [path[j][0] for j in range(max(0, i-window_size), min(len(path), i+window_size))]
        x_vals = [path[j][1] for j in range(max(0, i-window_size), min(len(path), i+window_size))]
        smoothed.append((np.mean(y_vals), np.mean(x_vals)))
    return smoothed

def robot_rect_points(center, orientation_rad, length_cells, width_cells):
    cx, cy = center[1], center[0]
    L = length_cells / 2
    W = width_cells / 2

    corners = [ (+L, +W), (+L, -W), (-L, -W), (-L, +W) ]
    cos_o = math.cos(orientation_rad)
    sin_o = math.sin(orientation_rad)

    rotated = []
    for dx, dy in corners:
        rx = cx + dx*cos_o - dy*sin_o
        ry = cy + dx*sin_o + dy*cos_o
        rotated.append((ry, rx))
    return rotated

# ----------- CONFIGURACIÓN ----------------
grid_size = (50, 70)
robot_length = 5
robot_width = 3
robot_radius = max(robot_length, robot_width) // 2
start = (2, 2)
end = (45, 65)

maze = np.zeros(grid_size, dtype=np.uint8)
maze[5:15, 10:25] = 1
maze[20:35, 5:20] = 1
maze[20:35, 30:50] = 1
maze[40:48, 25:55] = 1

maze_inflated = inflate_obstacles(maze, robot_radius)
if maze_inflated[end[0], end[1]] == 1:
    end = find_closest_free_cell(maze_inflated, end)

obstacles_points = [ ((7, 12), (7, 23)), ((25, 7), (33, 15)), ((25, 35), (33, 45)), ((43, 30), (46, 52)) ]
waypoints = [find_closest_free_cell(maze_inflated, pt) for pair in obstacles_points for pt in pair]
ordered_waypoints = find_closest_ordered_points(start, waypoints)

full_path = []
points_to_visit = [start] + ordered_waypoints + [end]
for i in range(len(points_to_visit)-1):
    segment = astar(maze_inflated, points_to_visit[i], points_to_visit[i+1])
    if not segment:
        print(f"No se pudo encontrar camino entre {points_to_visit[i]} y {points_to_visit[i+1]}")
        exit()
    if i > 0:
        segment = segment[1:]
    full_path.extend(segment)

full_path = smooth_path(full_path, max_angle_change_deg=25)
interp_path = interpolate_path(full_path, step=0.3)
interp_path = smooth_interpolated_path(interp_path, window_size=4)

# ----------- VISUALIZACIÓN ----------------
cell_size = 15
canvas = np.ones((grid_size[0]*cell_size, grid_size[1]*cell_size, 3), dtype=np.uint8) * 255

def draw_maze(canvas):
    for y in range(grid_size[0]):
        for x in range(grid_size[1]):
            if maze[y, x] == 1:
                cv2.rectangle(canvas, (x*cell_size, y*cell_size), ((x+1)*cell_size, (y+1)*cell_size), (0,0,0), -1)
            elif maze_inflated[y, x] == 1:
                cv2.rectangle(canvas, (x*cell_size, y*cell_size), ((x+1)*cell_size, (y+1)*cell_size), (180,180,180), -1)

def draw_points(canvas, points, color, radius=6):
    for p in points:
        cv2.circle(canvas, (int(p[1]*cell_size + cell_size//2), int(p[0]*cell_size + cell_size//2)), radius, color, -1)

draw_maze(canvas)
draw_points(canvas, [start], (0, 255, 0), radius=10)
draw_points(canvas, [end], (0, 0, 255), radius=10)
draw_points(canvas, waypoints, (255, 0, 0), radius=7)

cv2.namedWindow("Simulación Robot", cv2.WINDOW_NORMAL)

for i in range(len(interp_path)-1):
    frame = canvas.copy()
    p = interp_path[i]
    p_next = interp_path[min(i+1, len(interp_path)-1)]
    angle = math.atan2(p_next[0]-p[0], p_next[1]-p[1])
    rect = robot_rect_points(p, angle, robot_length, robot_width)
    pts = np.array([[int(x*cell_size), int(y*cell_size)] for y,x in rect], np.int32)
    cv2.polylines(frame, [pts], isClosed=True, color=(0,255,0), thickness=3)

    velocity = math.dist(p, p_next) / 0.05  # 50 ms por frame
    cv2.putText(frame, f"Vel: {velocity:.2f} celdas/s", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)

    cv2.imshow("Simulación Robot", frame)
    key = cv2.waitKey(50)
    if key == 27:
        break

cv2.destroyAllWindows()
