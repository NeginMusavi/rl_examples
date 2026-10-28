"""Physical collision, sensing, and conservative map connectivity."""

from collections import deque
import numpy as np

WORLD_SIZE = 10.0
ROBOT_RADIUS = 0.20
LIDAR_RANGE = 5.0
N_BEAMS = 36


def ray_circle_distance(origin, direction, circle):
    cx, cy, radius = circle
    oc = origin - np.array([cx, cy])
    b = np.dot(oc, direction)
    c = np.dot(oc, oc) - radius**2
    discriminant = b * b - c
    if discriminant < 0:
        return np.inf
    root = np.sqrt(discriminant)
    hits = [-b - root, -b + root]
    return min((t for t in hits if t >= 0), default=np.inf)


def ray_rectangle_distance(origin, direction, rect):
    x, y, w, h = rect
    low = np.array([x, y])
    high = low + [w, h]
    t_min, t_max = -np.inf, np.inf
    for k in range(2):
        if abs(direction[k]) < 1e-12:
            if origin[k] < low[k] or origin[k] > high[k]:
                return np.inf
        else:
            a = (low[k] - origin[k]) / direction[k]
            b = (high[k] - origin[k]) / direction[k]
            t_min = max(t_min, min(a, b))
            t_max = min(t_max, max(a, b))
    if t_max < max(t_min, 0):
        return np.inf
    return t_min if t_min >= 0 else t_max


def world_collision(pose, circles, rectangles, robot_radius=ROBOT_RADIUS):
    x, y = pose[:2]
    if min(x, y, WORLD_SIZE - x, WORLD_SIZE - y) <= robot_radius:
        return True
    for cx, cy, radius in circles:
        if np.hypot(x - cx, y - cy) <= radius + robot_radius:
            return True
    for rx, ry, w, h in rectangles:
        qx, qy = np.clip(x, rx, rx + w), np.clip(y, ry, ry + h)
        if np.hypot(x - qx, y - qy) <= robot_radius:
            return True
    return False


def inflate_obstacles(circles, rectangles, margin):
    return (
        [(x, y, r + margin) for x, y, r in circles],
        [
            (x - margin, y - margin, w + 2 * margin, h + 2 * margin)
            for x, y, w, h in rectangles
        ],
    )


def world_scan(pose, circles, rectangles, rng, noise_std=0.02, margin=0.0):
    perceived_circles, perceived_rectangles = inflate_obstacles(
        circles, rectangles, margin
    )
    origin = np.asarray(pose[:2], dtype=float)
    x, y = origin
    inside = any(np.hypot(x - cx, y - cy) <= r for cx, cy, r in perceived_circles)
    inside |= any(
        rx <= x <= rx + w and ry <= y <= ry + h for rx, ry, w, h in perceived_rectangles
    )
    angles = pose[2] + np.linspace(-np.pi, np.pi, N_BEAMS, endpoint=False)
    readings = []
    for angle in angles:
        direction = np.array([np.cos(angle), np.sin(angle)])
        hits = [ray_circle_distance(origin, direction, c) for c in perceived_circles]
        hits += [
            ray_rectangle_distance(origin, direction, r) for r in perceived_rectangles
        ]
        hits.append(
            ray_rectangle_distance(origin, direction, (0, 0, WORLD_SIZE, WORLD_SIZE))
        )
        # Consume the same number of random draws for every margin.
        noise = rng.normal(0, noise_std)
        readings.append(0.0 if inside else np.clip(min(hits) + noise, 0, LIDAR_RANGE))
    return np.asarray(readings, dtype=float)


def reachable(start, goal, circles, rectangles, resolution=0.25):
    """Conservative connectivity test using entirely free square grid cells.

    Inflate the robot by a cell's half diagonal. Every point in an accepted
    cell is then safe for the actual robot. Four-neighbor cells share an edge,
    and start/goal lie in their accepted cells. This can reject narrow but
    feasible paths; it cannot certify controller trackability.
    """
    if resolution <= 0:
        raise ValueError("resolution must be positive")
    n = int(np.ceil(WORLD_SIZE / resolution))
    width = WORLD_SIZE / n
    clearance = ROBOT_RADIUS + width / np.sqrt(2)

    def index(p):
        return tuple(np.clip((np.asarray(p) / width).astype(int), 0, n - 1))

    if world_collision([*start, 0], circles, rectangles):
        return False
    if world_collision([*goal, 0], circles, rectangles):
        return False
    s, g = index(start), index(goal)
    cache = {}

    def free(ij):
        if ij not in cache:
            center = [(ij[0] + 0.5) * width, (ij[1] + 0.5) * width, 0]
            cache[ij] = not world_collision(center, circles, rectangles, clearance)
        return cache[ij]

    if not free(s) or not free(g):
        return False
    queue, visited = deque([s]), {s}
    while queue:
        a, b = queue.popleft()
        if (a, b) == g:
            return True
        for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (a + da, b + db)
            if 0 <= nxt[0] < n and 0 <= nxt[1] < n and nxt not in visited and free(nxt):
                visited.add(nxt)
                queue.append(nxt)
    return False
