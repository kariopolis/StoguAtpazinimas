import numpy as np

from shapely.geometry import Polygon
from config import reference_points_num, margin

def generate_points_inside_contour(contour):

    #Contour flattening for better data reach
    poly = Polygon(contour.squeeze())

    # Compute inward offset distance
    offset = margin * np.sqrt(poly.area)
    inner_poly = poly.buffer(-offset)  # inward shrink

    # Generate  evenly spaced points along the inner polygon’s boundary
    points = [inner_poly.exterior.interpolate(i / reference_points_num, normalized=True).coords[0]
            for i in range(reference_points_num)]
    return(points)