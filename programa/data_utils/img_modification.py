import numpy as np
import cv2

from .file_utils import get_img

def get_mask(path):
    l_bound=np.array([199, 190, 170], dtype = np.uint8)
    #l_bound=np.array([80, 80, 70], dtype = np.uint8)
    u_bound=np.array([199, 195, 175], dtype = np.uint8)
    mask = cv2.inRange(cv2.cvtColor(get_img(path), cv2.COLOR_BGR2RGB), l_bound, u_bound)
    mask_bw = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  # 3 channels for saving consistency
    mask_gray = cv2.cvtColor(mask_bw, cv2.COLOR_BGR2GRAY)
    return mask_gray

def crop_image(img, base_x, base_y, size):
     return img[(base_x-int(size/2)):(base_x+int(size/2)),(base_y-int(size/2)):(base_y+int(size/2))]

def crop_image_according_to_mask(polygon, image):

    polygon = np.array(polygon, dtype=np.int32)
    mask = np.zeros(image.shape[:2], np.uint8)
    cv2.drawContours(mask, [polygon], -1, (255, 255, 255), -1, cv2.LINE_AA)
    cropped_image_according_to_mask = cv2.bitwise_and(image, image, mask=mask)
    
    return cropped_image_according_to_mask


