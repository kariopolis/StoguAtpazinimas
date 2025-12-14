import numpy as np
import cv2
import matplotlib.pyplot as plt
from config import pixel_size

def estimate_roof_angle(roof_mask):
    """
    Returns roof dominant angle in radians
    """
    ys, xs = np.where(roof_mask > 0)
    pts = np.column_stack([xs, ys]).astype(np.float32)

    mean = pts.mean(axis=0)
    pts -= mean

    cov = np.cov(pts.T)
    eigvals, eigvecs = np.linalg.eig(cov)

    main_vec = eigvecs[:, np.argmax(eigvals)]
    angle = np.arctan2(main_vec[1], main_vec[0])

    return angle

def build_roof_aligned_rectangles(obstacle_edges, roof_mask):
    roof_angle = estimate_roof_angle(roof_mask)

    c, s = np.cos(-roof_angle), np.sin(-roof_angle)
    R = np.array([[c, -s], [s, c]])

    c2, s2 = np.cos(roof_angle), np.sin(roof_angle)
    R_inv = np.array([[c2, -s2], [s2, c2]])

    rectangles = []

    for cnt in obstacle_edges:
        if len(cnt) < 5:
            continue

        pts = cnt.reshape(-1, 2).astype(np.float32)
        center = pts.mean(axis=0)

        pts_rot = (R @ (pts - center).T).T
        x, y, w, h = cv2.boundingRect(pts_rot.astype(np.int32))

        if w * h < 20:   # remove tiny noise
            continue

        box = np.array([
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h]
        ], dtype=np.float32)

        box = (R_inv @ box.T).T + center
        rectangles.append(box.astype(np.int32))

    return rectangles

def rect_distance(rect1, rect2):
    c1 = rect1.reshape(-1, 1, 2).astype(np.float32)
    c2 = rect2.reshape(-1, 1, 2).astype(np.float32)

    dmin = np.inf
    for p in c1:
        d = abs(cv2.pointPolygonTest(c2, tuple(p[0]), True))
        dmin = min(dmin, d)
    for p in c2:
        d = abs(cv2.pointPolygonTest(c1, tuple(p[0]), True))
        dmin = min(dmin, d)

    return dmin

def merge_close_rectangles(rectangles, roof_mask,
                           pixel_size_m=0.131,
                           merge_dist_m=1.1):
    """
    rectangles: list of Nx4x2 boxes (int)
    returns: merged rectangles, roof-aligned
    """

    roof_angle = estimate_roof_angle(roof_mask)  # radians

    # --- stable rotation center = roof centroid ---
    ys, xs = np.where(roof_mask > 0)
    center = np.array([xs.mean(), ys.mean()], dtype=np.float32)

    # rotation matrices
    c, s = np.cos(-roof_angle), np.sin(-roof_angle)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)

    c2, s2 = np.cos(roof_angle), np.sin(roof_angle)
    R_inv = np.array([[c2, -s2], [s2, c2]], dtype=np.float32)

    merge_dist_px = merge_dist_m / pixel_size_m

    # -------------------------------------------------
    # 1) Convert rectangles to roof-aligned AABBs
    # -------------------------------------------------
    aabbs = []
    for box in rectangles:
        pts = box.astype(np.float32)
        pts_r = (R @ (pts - center).T).T

        x1, y1 = pts_r.min(axis=0)
        x2, y2 = pts_r.max(axis=0)
        aabbs.append([x1, y1, x2, y2])

    # -------------------------------------------------
    # 2) Group AABBs by distance
    # -------------------------------------------------
    def aabb_distance(a, b):
        dx = max(0, max(a[0] - b[2], b[0] - a[2]))
        dy = max(0, max(a[1] - b[3], b[1] - a[3]))
        return np.hypot(dx, dy)

    N = len(aabbs)
    visited = [False] * N
    groups = []

    for i in range(N):
        if visited[i]:
            continue

        stack = [i]
        visited[i] = True
        group = [i]

        while stack:
            a = stack.pop()
            for b in range(N):
                if visited[b]:
                    continue
                if aabb_distance(aabbs[a], aabbs[b]) < merge_dist_px:
                    visited[b] = True
                    stack.append(b)
                    group.append(b)

        groups.append(group)

    # -------------------------------------------------
    # 3) Merge groups in roof space
    # -------------------------------------------------
    merged_rects = []

    for g in groups:
        xs = [aabbs[i][0] for i in g] + [aabbs[i][2] for i in g]
        ys = [aabbs[i][1] for i in g] + [aabbs[i][3] for i in g]

        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)

        box_r = np.array([
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2]
        ], dtype=np.float32)

        # rotate back
        box = (R_inv @ box_r.T).T + center
        merged_rects.append(box.astype(np.int32))

    return merged_rects





def roof_elements_detection(image, debug=True):
    """
    PURE EDGE-BASED obstacle detection.

    Uses ONLY edge geometry:
    - inside roof mask
    - away from roof boundary
    - rejects long straight edges

    Returns:
        obstacle_edge_contours
    """

    H, W = image.shape[:2]

    # Roof mask
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, roof_mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    roof_mask = roof_mask.astype(np.uint8)

    # Edge detection inside roof
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 40, 160)
    edges = cv2.bitwise_and(edges, roof_mask)

    # Distance to roof boundary
    roof_boundary = cv2.Canny(roof_mask, 50, 100)
    dist_to_boundary = cv2.distanceTransform(cv2.bitwise_not(roof_boundary),cv2.DIST_L2,5)

    # remove edges too close to roof boundary
    edges[dist_to_boundary < 6] = 0

    # Extract edge contours
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    obstacle_edges = []
    rejected = {"length": 0, "straight": 0}

    for cnt in contours:
        length = cv2.arcLength(cnt, False)

        # reject very long edges
        if length > 180:
            rejected["length"] += 1
            continue

        # straightness test. fit line and measure deviation
        pts = cnt.reshape(-1, 2).astype(np.float32)
        vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)

        # distance of points to fitted line
        dists = np.abs((vy * (pts[:, 0] - x0)) -(vx * (pts[:, 1] - y0)))

        curvature = np.mean(dists)

        # reject straight edges
        if curvature < 1.2 and length > 50:
            rejected["straight"] += 1
            continue


        obstacle_edges.append(cnt)

    # visualization
    if debug:
        print(
            f"[debug] edge contours={len(contours)} "
            f"obstacle_edges={len(obstacle_edges)} "
            f"rejected={rejected}"
        )

        dbg = image.copy()
        cv2.drawContours(dbg, obstacle_edges, -1, (0, 255, 0), 2)

        plt.figure(figsize=(14, 4))

        plt.subplot(1, 3, 1)
        plt.imshow(roof_mask, cmap="gray")
        plt.title("roof_mask")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(edges, cmap="gray")
        plt.title("filtered edges")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(dbg)
        plt.title("obstacle edges")
        plt.axis("off")

        plt.tight_layout()
        plt.show()

    rectangles = build_roof_aligned_rectangles(obstacle_edges,roof_mask)

    merged_rectangles = merge_close_rectangles(rectangles,roof_mask,pixel_size,merge_dist_m=1.1)

    dbg = image.copy()
    # create overlay
    overlay = dbg.copy()

    # draw filled RED blobs on overlay
    for box in merged_rectangles:
        cv2.drawContours(
            overlay,
            [box],
            contourIdx=0,
            color=(255, 0, 0),  # RED (RGB)
            thickness=-1        # FILLED
        )

    # alpha blend
    alpha = 0.45
    dbg = cv2.addWeighted(
        overlay, alpha,
        dbg, 1 - alpha,
        0
    )

    plt.figure(figsize=(6, 6))
    plt.imshow(dbg)
    plt.title("Merged obstacle exclusion zones (panel-aware)")
    plt.axis("off")
    plt.show()




    return obstacle_edges
