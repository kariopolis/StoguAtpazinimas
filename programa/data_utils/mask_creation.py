import cv2
import os
import numpy as np

from .file_utils import get_path, save_img
from .img_modification import get_mask, get_img, crop_image
from .visualization import see_img

from config import main_data_path as main, working_img_size as size

def create_mask(data_img_id=0):

    #Getting map image with polygons in it
    mask_iamges = get_path(os.path.join(main, "Images"), basename='Polygons')
    mask_path = mask_iamges[data_img_id]

    #Getting map image with orthophoto in it
    ortho_images=get_path(os.path.join(main, "Images"), basename='Ortho')
    ortho_path = ortho_images[data_img_id]
    
    mask = get_mask(mask_path)
    img = get_img(ortho_path)

    print(mask.shape, img.dtype)

    #Getting area of interest
    see_img(img, mask=False)
    print ('Input the center coordinates of the house')
    x=input('x: ')
    y=input('y: ')
    print(mask.shape)
    #mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)  
    binary_mask=crop_image(mask,int(y),int(x), size)
    img=crop_image(img, int(y),int(x), size)

    see_img(img, mask=False)
    see_img(binary_mask)
    
    #New data saving 
    answer=input('Do you want to save the data? Yes or No?').lower()
    while True:
        if answer== 'yes':
            masks = get_path(os.path.join(main, "Modified"), basename='Mask')
            imgs = get_path(os.path.join(main, "Modified"), basename='Ortho')
            save_img(binary_mask, path = os.path.join(main, r"Modified\Mask"), filename="Mask_"+str(len(masks)+1)+".png" )
            save_img(img, path = os.path.join(main, r"Modified\Ortho"), filename="Img_"+str(len(imgs)+1)+".png")
            break
        elif answer == 'no':
            print("The data was deleted.")
            break
        else:
            answer = input("Wrong input. Please repeat your answer. Do you want to save the data that was created? Yes or No?")
    

    return binary_mask, img, len(masks)+1

def contour_to_mask(contour, image = None, shape=None):
        contour = np.array(contour, dtype=np.int32).reshape(-1, 1, 2)
        if shape is None:
            shape = image.shape[:2]
        mask = np.zeros(shape, dtype=np.uint8)
        cv2.fillPoly(mask, [contour.reshape(-1, 1, 2).astype(np.int32)], 255)
        return mask

  
   
