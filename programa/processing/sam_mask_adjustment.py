import numpy as np
import cv2
import matplotlib.pyplot as plt

from .proc_utils import IoU, calculate_centroid
from .proc_utils import offset_line_point_direction, intersect_lines, compute_edge_normals, compute_clearance_along_normal, unit_vector_and_length
from data_utils import contour_to_mask, see_img


def keep_largest_connected_component(mask):
    mask_bin = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)
    if num_labels <= 1:
        return mask_bin * 255
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = 1 + np.argmax(areas)
    cleaned = np.zeros_like(mask_bin)
    cleaned[labels == largest_idx] = 1
    return cleaned * 255

def mask_adjustment(predicted_mask, scaled_mask, contour, image, visualize=False):

    #Making sure that both masks are the same shape
    predicted_mask_uint8 = ((predicted_mask > 0) * 255).astype(np.uint8)
    scaled_mask_uint8 = ((scaled_mask > 0) * 255).astype(np.uint8)

    #Making sure that predicted SAM mask has only one real contour and removing the noise
    predicted_mask_uint8_clean = keep_largest_connected_component(predicted_mask_uint8)

    #Calculating reference maxIoU for which the mask should be improved
    maxIoU = IoU(predicted_mask_uint8_clean, scaled_mask_uint8)

    #Getting the number of edges in the contour
    contour = np.array(contour, dtype=np.float32).reshape(-1, 2)
    num_of_edges=len(contour)

    current_contour = contour.copy()

    #Contour tuning phases. Contour tunning is being done in two phases: coarse and fine
    phases = [
        {"push_step": 1.0, "max_pushes": 200},  
        {"push_step": 0.25, "max_pushes": 400}  
    ]

    min_edge_length = 1e-3

    # for phase in phases:
    #     push_step = phase["push_step"]
    #     max_pushes_per_edge = phase["max_pushes"]
    #     for edge in range(num_of_edges):
    #         for direction_sign in [1.0, -1.0]:
    #             push_count = 0
    #             while push_count < max_pushes_per_edge:

    #                 center_now = calculate_centroid(current_contour)
    #                 normals, midpoints = compute_edge_normals(current_contour, center_now)

    #                 mid = midpoints[edge]
    #                 normal = normals[edge]

    #                 if np.linalg.norm(normal) < 1e-9:
    #                     p1 = current_contour[edge]
    #                     p2 = current_contour[(edge+1) % num_of_edges]
    #                     edge_direction, edge_length = unit_vector_and_length(p1, p2)
    #                     if edge_length < min_edge_length:
    #                         normal = np.array([0.0, 0.0])
    #                     else:
    #                         normal = np.array([-edge_direction[1], edge_direction[0]])
    #                         if np.dot(mid - center_now, normal) < 0:
    #                             normal = -normal
    #                 if np.linalg.norm(normal) < 1e-9:
    #                     break
                    
    #                 # STEP 1 | choosing the direction of the push
    #                 #check both outward push when direction_sign=1 and inward push when direction_sign=-1
    #                 check_normal = normal * direction_sign
                    
    #                 # STEP 2 | making sure that the push won't overshoot the boundaries of the image and reduce it if it does
    #                 #avoid invalid push over picture boundaries and make sure mid point moves after each push (important during fine tuning)
    #                 clearance_px = compute_clearance_along_normal(mid, check_normal, scaled_mask_uint8, max_check=120, step=1.0)
    #                 if clearance_px < 0.5:
    #                     break

    #                 #make sure to not overshoot the boundaries. Take smaller step if clearance is smaller than step
    #                 actual_push_size = min(push_step, clearance_px)
    #                 offset_pts = []
    #                 offset_directions =[]

    #                 # STEP 3 | creating a parallel line with applied offset
    #                 for edge_index in range(num_of_edges):

    #                     #avoid overshooting the list, when dealing with the last value
    #                     p1 = current_contour[edge_index]
    #                     p2 = current_contour[(edge_index + 1) % num_of_edges]

    #                     #calculate actual offset 
    #                     offset = direction_sign * actual_push_size if edge_index == edge else 0.0
    #                     edge_normal = normals[edge_index]

    #                     #make sure edge's normal vector is not too short
    #                     if np.linalg.norm(edge_normal) < 1e-15:
    #                         edge_direction, edge_length = unit_vector_and_length(p1, p2)
    #                         if edge_length < min_edge_length:
    #                             edge_normal = np.array([0.0, 0.0])
    #                         else:
    #                             edge_normal = np.array([-edge_direction[1], edge_direction[0]])
    #                             if np.dot(midpoints[edge_index] - center_now, edge_normal) < 0:
    #                                 edge_normal = -edge_normal


    #                     #computing the line parallel to the edge, but with applied offset and storing the value
    #                     point_on_line, direction_unit = offset_line_point_direction(p1, p2, edge_normal, offset)
    #                     offset_pts.append(point_on_line)
    #                     offset_directions.append(direction_unit)

    #                 # STEP 4 | 
    #                 new_vertices = []
    #                 for v_idx in range(num_of_edges):
    #                     prev_idx = (v_idx - 1) % num_of_edges
    #                     inter = intersect_lines(offset_pts[prev_idx], offset_directions[prev_idx],
    #                                             offset_pts[v_idx], offset_directions[v_idx])
    #                     if inter is None:
    #                         v_orig = current_contour[v_idx]
    #                         nA = normals[prev_idx]
    #                         nB = normals[v_idx]
    #                         avg_n = nA + nB
    #                         inter = v_orig if np.linalg.norm(avg_n) < 1e-9 else v_orig + (avg_n / np.linalg.norm(avg_n)) * 0.1

    #                     new_vertices.append(inter)
    #                 new_vertices = np.array(new_vertices, dtype=np.float32)
    #                 if np.any(~np.isfinite(new_vertices)):
    #                     break
                    
    #                 candidate_mask = contour_to_mask(new_vertices, shape = predicted_mask.shape)

    #                 candidate_IoU = IoU(predicted_mask_uint8_clean, candidate_mask)

    #                 if (candidate_IoU > maxIoU + 1e-9) or abs(candidate_IoU - maxIoU) < 1e-9:
    #                     maxIoU = max(maxIoU, candidate_IoU)
    #                     current_contour = new_vertices
    #                     push_count += 1
    #                 else:
    #                     break

    for phase in phases:
        push_step = phase["push_step"]

        while True:  
            improvement_made = False

            # Try each wall ONCE per cycle
            for edge in range(num_of_edges):

                best_local_IoU = maxIoU
                best_local_contour = None

                # Test: push outward (direction_sign = +1) and inward (direction_sign = -1)
                for direction_sign in [1.0, -1.0]:

                    center_now = calculate_centroid(current_contour)
                    normals, midpoints = compute_edge_normals(current_contour, center_now)
                    mid = midpoints[edge]
                    normal = normals[edge]

                    # Fix degenerate normals
                    if np.linalg.norm(normal) < 1e-9:
                        p1 = current_contour[edge]
                        p2 = current_contour[(edge + 1) % num_of_edges]
                        edge_direction, edge_length = unit_vector_and_length(p1, p2)
                        if edge_length < min_edge_length:
                            continue
                        normal = np.array([-edge_direction[1], edge_direction[0]])
                        if np.dot(mid - center_now, normal) < 0:
                            normal = -normal

                    if np.linalg.norm(normal) < 1e-9:
                        continue

                    check_normal = normal * direction_sign
                    clearance_px = compute_clearance_along_normal(mid, check_normal,
                                                                scaled_mask_uint8, max_check=120, step=1.0)
                    if clearance_px < 0.5:
                        continue

                    actual_push_size = min(push_step, clearance_px)
                    offset_pts = []
                    offset_dirs = []

                    # Build new candidate polygon
                    for edge_idx in range(num_of_edges):
                        p1 = current_contour[edge_idx]
                        p2 = current_contour[(edge_idx + 1) % num_of_edges]

                        offset = actual_push_size * direction_sign if edge_idx == edge else 0.0
                        edge_normal = normals[edge_idx]

                        if np.linalg.norm(edge_normal) < 1e-15:
                            edge_direction, edge_length = unit_vector_and_length(p1, p2)
                            if edge_length < min_edge_length:
                                edge_normal = np.array([0.0, 0.0])
                            else:
                                edge_normal = np.array([-edge_direction[1], edge_direction[0]])
                                if np.dot(midpoints[edge_idx] - center_now, edge_normal) < 0:
                                    edge_normal = -edge_normal

                        pt, d = offset_line_point_direction(p1, p2, edge_normal, offset)
                        offset_pts.append(pt)
                        offset_dirs.append(d)

                    new_vertices = []
                    for v_idx in range(num_of_edges):
                        prev_idx = (v_idx - 1) % num_of_edges
                        inter = intersect_lines(offset_pts[prev_idx], offset_dirs[prev_idx],
                                                offset_pts[v_idx], offset_dirs[v_idx])
                        if inter is None:
                            v_orig = current_contour[v_idx]
                            avg_n = normals[prev_idx] + normals[v_idx]
                            if np.linalg.norm(avg_n) > 1e-9:
                                v_orig = v_orig + (avg_n / np.linalg.norm(avg_n)) * 0.1
                            inter = v_orig
                        new_vertices.append(inter)

                    new_vertices = np.array(new_vertices, dtype=np.float32)
                    if np.any(~np.isfinite(new_vertices)):
                        continue

                    candidate_mask = contour_to_mask(new_vertices, shape=predicted_mask.shape)
                    candidate_IoU = IoU(predicted_mask_uint8_clean, candidate_mask)

                    if candidate_IoU > best_local_IoU:
                        best_local_IoU = candidate_IoU
                        best_local_contour = new_vertices

                # After testing both push & pull → apply best option IF it helps
                if best_local_contour is not None:
                    current_contour = best_local_contour
                    maxIoU = best_local_IoU
                    improvement_made = True

            # End of cycle — no improvement → stop this phase
            if not improvement_made:
                break


    best_contour = current_contour.reshape(-1, 2)
    best_mask = contour_to_mask(best_contour, shape = predicted_mask.shape)

    # Visualization
    if visualize:
        def show_mask(mask, ax, color=(0, 1, 0, 0.45)):
            h, w = mask.shape
            mask_image = (mask.reshape(h, w, 1) / 255.0) * np.array(color).reshape(1, 1, -1)
            ax.imshow(mask_image)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        show_mask(predicted_mask, plt.gca())
        plt.title("Initial SAM Prediction")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(image)
        show_mask(best_mask, plt.gca())
        plt.title(f"Optimized Mask (IoU: {maxIoU:.4f})")
        plt.axis('off')
        plt.show()

    return best_mask, best_contour

