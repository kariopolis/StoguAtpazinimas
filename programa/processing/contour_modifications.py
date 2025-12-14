import cv2
import math
import numpy as np

from .proc_utils import calculate_centroid, merge_close_points, unit_vector_and_length
from data_utils import contour_to_mask, see_img

def finding_closest_contour(mask):

    #Finding all the contours in the mask
    contours,_= cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    if contours:
        # Find the center contour
        height, width = mask.shape
        center_x, center_y = width // 2, height // 2
        min_distance = float('inf')
        closest_contour = None
        for contour in contours:
            centroid = calculate_centroid(contour)
            if centroid is None: continue
            distance = np.sqrt((centroid[0] - center_x)**2 + (centroid[1] - center_y)**2)
            if distance < min_distance:
                min_distance = distance
                closest_contour = contour

        #Creating an img only with the closest contour
        if closest_contour is not None:
            selected_contours = [closest_contour]
            new_mask = contour_to_mask(selected_contours, np.array(mask))
        else:
            print("No valid center contour found")
            return None
    else:
        print("No contours found in the mask")
        return None
    
    return new_mask, closest_contour


# def get_polygon_corners_from_contour(contour, angle_threshold, simplify_epsilon):
#         if simplify_epsilon > 0:
#             contour = cv2.approxPolyDP(contour, simplify_epsilon, True)

#         contour = contour[:, 0, :]
#         corners = []
#         for i in range(len(contour)):
            
#             # Three points for to determine 2 vectors
#             p_prev = contour[i - 1]
#             p_curr = contour[i]
#             p_next = contour[(i + 1) % len(contour)]

#             # Computing vectors from current points and normalizing them
#             unit_v1,_=unit_vector_and_length(p_curr, p_prev)
#             unit_v2,_=unit_vector_and_length(p_curr, p_next)

#             #calculating the angle between vectors (cos*|v1|x|v2|)
#             angle = np.degrees(np.arccos(np.clip(np.dot(unit_v1, unit_v2), -1.0, 1.0)))

#             if angle < angle_threshold:
#                 corners.append(p_curr)
#         return np.array(corners, dtype=np.int32)

import numpy as np
import cv2

def unit_vector_and_length(p1, p2):
    v = p2 - p1
    length = np.linalg.norm(v)
    if length == 0:
        return v, 0
    return v / length, length


def merge_collinear_edges(contour, direction_threshold_deg=10):
    """
    Merge consecutive edges whose direction is similar.
    Returns a simplified contour.
    """
    pts = contour.reshape(-1, 2)
    merged_pts = [pts[0]]

    prev_vec = None
    
    for i in range(1, len(pts)+1):
        p_prev = merged_pts[-1]
        p_curr = pts[i % len(pts)]
        
        vec, _ = unit_vector_and_length(p_prev, p_curr)
        
        if prev_vec is None:
            prev_vec = vec
            merged_pts.append(p_curr)
            continue

        # angle between current edge and previous
        angle = np.degrees(np.arccos(np.clip(np.dot(prev_vec, vec), -1, 1)))

        if angle < direction_threshold_deg:
            # instead of creating a new point, overwrite the last one
            # → merges jagged micro-edges
            merged_pts[-1] = p_curr
        else:
            merged_pts.append(p_curr)
            prev_vec = vec

    return np.array(merged_pts, dtype=np.int32).reshape(-1, 1, 2)


def get_polygon_corners_from_contour(contour, angle_threshold=150, simplify_epsilon=3,
                                     collinear_merge_angle=50):

    # 1) Optional RDP simplification
    if simplify_epsilon > 0:
        contour = cv2.approxPolyDP(contour, simplify_epsilon, True)

    # 2) Merge nearly-collinear edges
    contour = merge_collinear_edges(contour, collinear_merge_angle)

    pts = contour[:, 0, :]
    corners = []

    for i in range(len(pts)):
        p_prev = pts[i - 1]
        p_curr = pts[i]
        p_next = pts[(i + 1) % len(pts)]

        # 3) Compute normalized vectors
        v1, _ = unit_vector_and_length(p_curr, p_prev)
        v2, _ = unit_vector_and_length(p_curr, p_next)

        angle = np.degrees(np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)))

        # small angle = sharp corner
        if angle < angle_threshold:
            corners.append(p_curr)

    return np.array(corners, dtype=np.int32)



def scale_contour(contour, scale_factor, mask, angle_threshold=150, simplify_epsilon=1, merge_distance=6): #140 
    # Compute contour center
    centroid=calculate_centroid(contour)

    # Scale contour
    scaled_contour = (contour - centroid) * scale_factor + centroid
    scaled_contour = scaled_contour.reshape(-1, 1, 2).astype(np.float32)

    # Detect corners from contour 
    corners = get_polygon_corners_from_contour(scaled_contour, angle_threshold, simplify_epsilon)

    scaled_mask = contour_to_mask(corners, shape = mask)
    see_img (scaled_mask, "Scaled Contour with Filtered Corners")


    # merge nearby corner points
    corners = merge_close_points(corners, merge_distance)

    # Draw scaled mask contour and corners 
    scaled_mask = contour_to_mask(scaled_contour, shape = mask)

    # Draw scaled contour and corners on the mask
    vis = cv2.cvtColor(scaled_mask, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(vis, [corners.astype(np.int32)], -1, (0, 255, 0), 2)
    for c in corners:
         cv2.circle(vis, tuple(c), 4, (255, 0, 0), -1)
    see_img (vis, "Scaled Contour with Filtered Corners")


    return scaled_mask, corners