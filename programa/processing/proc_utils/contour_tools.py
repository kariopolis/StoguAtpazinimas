import cv2
import math
import numpy as np

from config import pixel_size

#Contours center of mass calculation
def calculate_centroid(contour):

    if contour is None or len(contour) == 0:
        raise ValueError("Empty contour provided.")
    contour_reshaped = contour.reshape(-1, 1, 2).astype(np.int32)
    M = cv2.moments(contour_reshaped)
    if M["m00"] != 0:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        return np.array([cx, cy], dtype=np.float64)
    else:
        pts = contour.reshape(-1, 2)
        return np.mean(pts, axis=0)
    

#Merge points to one if distance between them is too small
def merge_close_points(points, threshold):

    #The function adds a point to merge list if it is not too close to any other point in the merge list

    if len(points) == 0:
        return np.array([], dtype=np.int32)
    merged = []
    points = points.tolist()
    for point in points:
        keep = True
        for m in merged:
            dist = math.hypot(point[0] - m[0], point[1] - m[1])
            if dist < threshold:
                keep = False
                break
        if keep:
            merged.append(point)
    return np.array(merged, dtype=np.int32)

#Calculate unit vector and its length
def unit_vector_and_length(p1, p2):
    vector = p2 - p1
    length = np.linalg.norm(vector)

    #avoid noise
    if length < 1e-8:
        return np.array([0.0, 0.0]), 0.0
    
    #for vector direction calculations
    unit_vector = vector/length

    return unit_vector, length

#Compute normal vector of an edge vector
def compute_edge_normals(pts, center):
    num_of_points = len(pts)

    #Creating arrays to save normal vectors and midpoints of the edges
    normals = np.zeros((num_of_points, 2), dtype=np.float64)
    midpoints = np.zeros((num_of_points, 2), dtype=np.float64)

    for i in range(num_of_points):
        p1 = pts[i]
        p2 = pts[(i+1) % num_of_points]

        #Get normal vector(direction) and length of the edge
        edge_direction, length = unit_vector_and_length(p1, p2)

        #Skipping invalid edges
        if length == 0:
            normals[i] = np.array([0.0, 0.0])
            midpoints[i] = (p1 + p2) / 2
            continue
        
        #Turning edge vector by 90 degrees to the left to get a normal vector
        normal_vector = np.array([-edge_direction[1], edge_direction[0]])

        #Calculating midpoint of the edge
        mid = (p1 + p2) / 2
        midpoints[i] = mid

        #Checking if normal vector points inwards or outwards 
        if np.dot(mid - center, normal_vector) > 0:
            outward = normal_vector
        else:
            outward = -normal_vector
        normals[i] = outward
    return normals, midpoints

#Checks clearace for the push
def compute_clearance_along_normal(mid_pt, normal, ref_mask_uint8, max_check=120, step=1.0):
    # Steps from mid_pt along normal until the mask boundary changes.
    h, w = ref_mask_uint8.shape[:2]

    normal_length = np.linalg.norm(normal)
    if normal_length < 1e-9:
        return 0.0
    
    #Normalizing normal vector (values from 0 to 1)
    n = normal / normal_length

    def sample(px, py):
        intx, inty = int(round(px)), int(round(py))
        if intx < 0 or intx >= w or inty < 0 or inty >= h:
            return 0.0
        return ref_mask_uint8[inty, intx]

    x0, y0 = mid_pt
    mid_val = sample(x0, y0)
    d = step
    while d <= max_check:

        #Calculating midpoint coordinates after the push
        sx, sy = x0 + n[0] * d, y0 + n[1] * d

        #Rounding calculated coordinates
        val = sample(sx, sy)

        #Checking if rouned value after push doesn't give back the same midpoint pixel value as before the push
        if val != mid_val:
            return d
        
        #Making another push if rounded coordinates did not change after the push
        d += step
    return max_check

#Offset edge basepoint 
def offset_line_point_direction(p1, p2, normal, offset):
        
    dir_vec = p2 - p1
    dir_len = np.linalg.norm(dir_vec)
    if dir_len < 1e-8:
        return p1 + normal * offset, np.array([1.0, 0.0])
    dir_unit = dir_vec / dir_len
    base_pt = p1 + normal * offset
    return base_pt, dir_unit

#Two lines intersection point calculation
def intersect_lines(pt1, dir1, pt2, dir2):
    
    '''
        *   Line r1 = pt1 + t1 * dir1 
            Line r2 = pt2 + t2 * dir2 
        *   Lines intersect, where:  
                pt1 + t1 * dir1 = pt2 + t2 * dir2
        *   Moving all knowns to the right side:
                t1 * dir1 - t2 * dir2 = pt2 - pt1
                (dir1 - dir2) * (t1 + t2) = pt2 - pt1
        *   (dir1 - dir2) is coeficient A :
                A=|dir1x -dir2x|
                |dir1y -dir2y|
        *   (pt2 - pt1) - right-hands side (rhs) is a displacement vector from one lines point to the others
            Full equation:
                A x |t1|=rhs
                    |t2|
        *   To check if lines intersect and are not parallel we need to calculate determinant from A
        *   If determinant is equal to 0 or close to then lines are parallel or close to parallel and lines dont realy
            have an intersection.
        *   Solving an equation 
                |t1|=A**(-1) x rhs
                |t2|
                sol[0]=t1, sol[2]=t2
        *   
    '''
    #coefficient_matrix A
    A = np.column_stack((dir1, -dir2))
    #Right hand-side matrix 
    rhs = pt2 - pt1
    #Determiant calculation
    det = np.linalg.det(A)

    #Returm none if determianant is 0 or near it
    if abs(det) < 1e-9:
        return None
    #Solving an equation
    solution = np.linalg.solve(A, rhs)
    t1 = solution[0]
    
    #Calculating coordinates of intersection
    intersection = pt1 + t1 * dir1
    return intersection

def calculate_wall_measurements(polygon):
    #Walls start from the western one
    walls=[]
    for i in range(0,len(polygon)):
        if i!=len(polygon)-1:
            walls.append((np.sqrt((polygon[i][0]-polygon[i+1][0])**2+(polygon[i][1]-polygon[i+1][1])**2))*pixel_size)
        else:
            walls.append((np.sqrt((polygon[0][0]-polygon[-1][0])**2+(polygon[0][1]-polygon[-1][1])**2))*pixel_size)
    return walls 

def calculate_roof_area(contour):

    contour = np.asarray(contour, dtype=np.float64)

    x = contour[:, 0]
    y = contour[:, 1]

    #Shoelace formula
    area_pixels = 0.5 * abs(
        np.dot(x, np.roll(y, -1)) -
        np.dot(y, np.roll(x, -1))
    )

    #Convert pixels → real units
    area_real = area_pixels * (pixel_size ** 2)

    return area_real