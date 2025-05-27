import cv2
import numpy as np
import heapq
from collections import defaultdict, deque
from scipy.spatial.distance import cdist
from scipy.interpolate import splprep, splev

# =========================
# Algoritmo A* Mejorado
# =========================
class AStar:
    def __init__(self, grid, start, goal, penalty_map=None):
        self.grid = grid
        self.start = start
        self.goal = goal
        self.rows, self.cols = grid.shape
        self.penalty_map = penalty_map if penalty_map is not None else np.zeros_like(grid, dtype=np.float32)

    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def get_neighbors(self, node):
        neighbors = []
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0),
                      (1, 1), (-1, -1), (1, -1), (-1, 1)]
        for dx, dy in directions:
            x, y = node[0] + dx, node[1] + dy
            if (0 <= x < self.rows and 0 <= y < self.cols and 
                self.grid[x, y] == 0):
                neighbors.append((x, y))
        return neighbors

    def find_path(self):
        open_set = [(0, self.start)]
        came_from = {}
        g_score = defaultdict(lambda: float('inf'))
        g_score[self.start] = 0
        f_score = defaultdict(lambda: float('inf'))
        f_score[self.start] = self.heuristic(self.start, self.goal)

        while open_set:
            current = heapq.heappop(open_set)[1]
            if current == self.goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(self.start)
                return path[::-1]

            for neighbor in self.get_neighbors(current):
                penalty = self.penalty_map[neighbor[0], neighbor[1]]
                tentative_g_score = g_score[current] + 1 + penalty
                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, self.goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return []

# =========================
# Inflado de obstáculos con margen extra
# =========================
def inflate_obstacles(grid, real_width_cm, real_height_cm, robot_size_cm, margin_cm=10):
    grid_h, grid_w = grid.shape
    cell_width_cm = real_width_cm / grid_w
    cell_height_cm = real_height_cm / grid_h
    inflation_radius_x = int(np.ceil(((robot_size_cm + margin_cm) / 2) / cell_width_cm))
    inflation_radius_y = int(np.ceil(((robot_size_cm + margin_cm) / 2) / cell_height_cm))
    inflated_grid = grid.copy()
    for i in range(grid_h):
        for j in range(grid_w):
            if grid[i, j] == 1:
                for dx in range(-inflation_radius_y, inflation_radius_y + 1):
                    for dy in range(-inflation_radius_x, inflation_radius_x + 1):
                        ni, nj = i + dx, j + dy
                        if 0 <= ni < grid_h and 0 <= nj < grid_w:
                            inflated_grid[ni, nj] = 1
    return inflated_grid

# =========================
# Penalización por cercanía
# =========================
def create_penalty_map(grid, waypoints, penalty_radius=5, penalty_value=10):
    penalty_map = np.zeros_like(grid, dtype=np.float32)
    for wp in waypoints:
        for dr in range(-penalty_radius, penalty_radius + 1):
            for dc in range(-penalty_radius, penalty_radius + 1):
                r, c = wp[0] + dr, wp[1] + dc
                if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]:
                    dist = np.hypot(dr, dc)
                    if dist <= penalty_radius:
                        penalty_map[r, c] += penalty_value * (1 - dist / (penalty_radius + 1))
    return penalty_map

# =========================
# Interpolación spline para trayectorias curvas
# =========================
def spline_path(path, num_points=400):
    path = np.array(path)
    x = path[:,1]
    y = path[:,0]
    tck, u = splprep([x, y], s=3)
    unew = np.linspace(0, 1, num_points)
    out = splev(unew, tck)
    smooth = np.stack([out[1], out[0]], axis=1)  # (fila, columna)
    return [tuple(map(float, p)) for p in smooth]

# =========================
# Utilidades de visión y geometría
# =========================
def draw_real_scale_grid(img, real_width_cm=203, real_height_cm=403, lines_x=11, lines_y=11,
                         color=(200, 200, 200), thickness=1):
    h, w = img.shape[:2]
    step_x = w / (lines_x - 1)
    step_y = h / (lines_y - 1)
    step_x_cm = real_width_cm / (lines_x - 1)
    step_y_cm = real_height_cm / (lines_y - 1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    for i in range(lines_x):
        x = int(i * step_x)
        cv2.line(img, (x, 0), (x, h), color, thickness)
        text = f"{i * step_x_cm:.0f}cm"
        cv2.putText(img, text, (x + 2, 15), font, font_scale, color, 1)
    for j in range(lines_y):
        y = int(j * step_y)
        cv2.line(img, (0, y), (w, y), color, thickness)
        text = f"{(lines_y - 1 - j) * step_y_cm:.0f}cm"
        cv2.putText(img, text, (2, y - 5), font, font_scale, color, 1)

def detect_colors(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask_black = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 50]))
    mask_orange = cv2.inRange(hsv, np.array([5, 100, 100]), np.array([15, 255, 255]))
    mask_green = cv2.inRange(hsv, np.array([40, 60, 60]), np.array([70, 255, 255]))
    mask_obstacles = cv2.bitwise_or(mask_black, mask_orange)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask_obstacles = cv2.morphologyEx(mask_obstacles, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask_obstacles = cv2.morphologyEx(mask_obstacles, cv2.MORPH_OPEN, kernel_open, iterations=1)
    return mask_black, mask_orange, mask_green, mask_obstacles

def detect_black_rectangles(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rectangles = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 300:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
        if len(approx) < 4:
            continue
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        width, height = rect[1]
        if width == 0 or height == 0:
            continue
        aspect_ratio = max(width, height) / min(width, height)
        if aspect_ratio > 8.0:
            continue
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            continue
        solidity = float(area) / hull_area
        if solidity < 0.8:
            continue
        rectangles.append(box)
    return rectangles

def create_grid_from_mask(mask, grid_size=(60, 60)):
    h, w = mask.shape
    grid_h, grid_w = grid_size
    grid = np.zeros((grid_h, grid_w), dtype=np.uint8)
    scale_y = h / grid_h
    scale_x = w / grid_w
    for i in range(grid_h):
        for j in range(grid_w):
            y_start = int(i * scale_y)
            y_end = int((i + 1) * scale_y)
            x_start = int(j * scale_x)
            x_end = int((j + 1) * scale_x)
            area = mask[y_start:y_end, x_start:x_end]
            if np.sum(area > 0) > (area.size * 0.3):
                grid[i, j] = 1
    return grid, scale_x, scale_y

def grid_to_image_coords(grid_point, scale_x, scale_y):
    x = int(grid_point[1] * scale_x + scale_x / 2)
    y = int(grid_point[0] * scale_y + scale_y / 2)
    return (x, y)

def real_to_grid_coords(x_cm, y_cm, real_width_cm, real_height_cm, grid_size):
    grid_h, grid_w = grid_size
    row = int((real_height_cm - y_cm) / real_height_cm * grid_h)
    col = int(x_cm / real_width_cm * grid_w)
    return max(0, min(grid_h - 1, row)), max(0, min(grid_w - 1, col))

def find_nearest_free(grid, point):
    grid_h, grid_w = grid.shape
    visited = set()
    queue = deque([point])
    while queue:
        r, c = queue.popleft()
        if 0 <= r < grid_h and 0 <= c < grid_w:
            if grid[r, c] == 0:
                return (r, c)
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    nr, nc = r + dr, c + dc
                    if (nr, nc) not in visited:
                        visited.add((nr, nc))
                        queue.append((nr, nc))
    return point

def draw_robot_square(img, grid_point, scale_x, scale_y, robot_size_cm, real_width_cm, real_height_cm, color=(255,0,0)):
    center = grid_to_image_coords(grid_point, scale_x, scale_y)
    px_w = int(robot_size_cm * img.shape[1] / real_width_cm)
    px_h = int(robot_size_cm * img.shape[0] / real_height_cm)
    top_left = (int(center[0] - px_w/2), int(center[1] - px_h/2))
    bottom_right = (int(center[0] + px_w/2), int(center[1] + px_h/2))
    cv2.rectangle(img, top_left, bottom_right, color, 2)
    overlay = img.copy()
    cv2.rectangle(overlay, top_left, bottom_right, color, -1)
    alpha = 0.18
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

def get_obstacle_centroids_in_grid(rectangles, scale_x, scale_y, grid_size):
    centroids = []
    for rect in rectangles:
        M = cv2.moments(rect)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            grid_point = (int(cY / scale_y), int(cX / scale_x))
            centroids.append(grid_point)
    return centroids

def compute_angle_and_speed_cm(path, idx, scale_x, scale_y, real_width_cm, real_height_cm, tiempo_paso_s=0.2):
    if idx == 0 or idx >= len(path)-1:
        return 0, 10
    p_prev = np.array(grid_to_image_coords(path[idx-1], scale_x, scale_y))
    p_curr = np.array(grid_to_image_coords(path[idx], scale_x, scale_y))
    p_next = np.array(grid_to_image_coords(path[idx+1], scale_x, scale_y))
    v1 = p_curr - p_prev
    v2 = p_next - p_curr
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0, 10
    cos_angle = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
    angle = np.arccos(cos_angle) * 180 / np.pi
    dist_px = np.linalg.norm(p_next - p_curr)
    px_per_cm = (scale_x + scale_y) / 2
    dist_cm = dist_px / px_per_cm
    if angle > 60:
        vel = max(10, dist_cm / tiempo_paso_s)
    elif angle > 20:
        vel = max(20, dist_cm / tiempo_paso_s)
    else:
        vel = max(30, dist_cm / tiempo_paso_s)
    return angle, vel

# =========================
# MAIN LOOP con cámara
# =========================
real_width_cm = 203
real_height_cm = 403
robot_size_cm = 30
holgura_cm = 10
grid_size = (60, 60)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    mask_black, mask_orange, mask_green, mask_obstacles = detect_colors(frame)
    rectangles = detect_black_rectangles(mask_obstacles)
    grid, scale_x, scale_y = create_grid_from_mask(mask_obstacles, grid_size=grid_size)
    grid = inflate_obstacles(grid, real_width_cm, real_height_cm, robot_size_cm, margin_cm=holgura_cm)

    draw_real_scale_grid(frame, real_width_cm, real_height_cm, 11, 11)
    for rect in rectangles:
        cv2.drawContours(frame, [rect], -1, (0, 255, 0), 2)

    # Puedes ajustar estos puntos de inicio/fin según la posición real en la pista
    start_real = (183, 202)
    goal_real = (0, 202)
    start_grid = real_to_grid_coords(*start_real, real_width_cm, real_height_cm, grid_size)
    goal_grid = real_to_grid_coords(*goal_real, real_width_cm, real_height_cm, grid_size)
    start_grid = find_nearest_free(grid, start_grid)
    goal_grid = find_nearest_free(grid, goal_grid)

    waypoints = get_obstacle_centroids_in_grid(rectangles, scale_x, scale_y, grid_size)
    waypoints = [find_nearest_free(grid, wp) for wp in waypoints]

    waypoints_ordered = [start_grid]
    remaining = waypoints.copy()
    while remaining:
        dists = cdist([waypoints_ordered[-1]], remaining)
        idx = np.argmin(dists)
        waypoints_ordered.append(remaining.pop(idx))
    waypoints_ordered.append(goal_grid)

    penalty_map = create_penalty_map(grid, waypoints, penalty_radius=4, penalty_value=7)

    full_path = []
    for i in range(len(waypoints_ordered) - 1):
        astar_segment = AStar(grid, waypoints_ordered[i], waypoints_ordered[i + 1], penalty_map=penalty_map)
        segment_path = astar_segment.find_path()
        if not segment_path:
            continue
        if full_path and segment_path[0] == full_path[-1]:
            full_path.extend(segment_path[1:])
        else:
            full_path.extend(segment_path)

    if full_path:
        curved_path = spline_path(full_path, num_points=200)
        temp_img = frame.copy()
        for idx, pt in enumerate(curved_path):
            img_step = temp_img.copy()
            if idx > 0:
                pts = [grid_to_image_coords((p[0], p[1]), scale_x, scale_y) for p in curved_path[:idx+1]]
                cv2.polylines(img_step, [np.array(pts)], False, (0,255,255), 3)
            draw_robot_square(img_step, pt, scale_x, scale_y, robot_size_cm, real_width_cm, real_height_cm, color=(255,0,0))
            angle, vel = compute_angle_and_speed_cm(curved_path, idx, scale_x, scale_y, real_width_cm, real_height_cm, tiempo_paso_s=0.2)
            cv2.putText(img_step, f"Ángulo: {angle:.1f}°", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            cv2.putText(img_step, f"Velocidad: {vel:.1f} cm/s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            cv2.imshow('Recorrido paso a paso', img_step)
            key = cv2.waitKey(1)
            if key == 27:
                break
    else:
        cv2.putText(frame, "No se encontró camino completo", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow('Recorrido paso a paso', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
