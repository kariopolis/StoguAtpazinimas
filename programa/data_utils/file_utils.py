import os
import cv2

from config import main_data_path as main

def get_path(path, basename):
        file_path=[]
        for root,dirs,files in os.walk(path):
            #print (os.path.basename(root))
            if os.path.basename(root) == basename:
                #print ("found t")
                for file in files:
                    if ".png" in file:
                         file_path.append(os.path.join(root, file))
        return file_path

def get_img(path):
    img = cv2.imread(path)
    return img

def save_img(img, path, filename):
     
     cv2.imwrite(os.path.join(path,filename), img)
     print("Successfully saved to: "+os.path.join(path,filename))

def get_existing_mask(mask_id):
    try:
        mask_path = get_path(os.path.join(main, r"Modified"),"Mask")[mask_id]
    except (ValueError, IndexError, FileNotFoundError) as e:
        print(f"Mask ID {mask_id} is not valid or out of range: {e}")
        return None
    mask = get_img(mask_path)

    if len(mask.shape)==3 and mask.shape[2]>1:
        #Final mask processing to make sure tha it is a binary 1 channel image
        # convert to grayscale
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) 
        # create a binary mask 
        _, binary_mask = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)  

    binary_mask = binary_mask.astype('uint8')

    return binary_mask

def get_existing_img(img_id):
    try:
        img_path = get_path(os.path.join(main, r"Modified"),"Ortho")[img_id]
        return get_img(img_path)
    except (ValueError, IndexError, FileNotFoundError) as e:
        print(f"Image ID {img_id} is not valid or out of range: {e}")
        return None
    
def save_roof_img(image, id, basename="Ortho"):
    path = get_path(os.path.join(main, r"Modified"), basename)[id]
    saving_path=main+"/OnlyRoof"
    print("Roof image: roof_"+str(os.path.basename(path))+"\n")
    save_img(image, saving_path, "roof_"+str(os.path.basename(path)))

def save_wall_measurements(walls, id, basename="Ortho"):
    path = get_path(os.path.join(main, r"Modified"), basename)[id]
    file_path=main+"/OnlyRoof/roof_"+str(os.path.basename(path))
    file_path = file_path.replace('png', 'txt')
    with open(file_path, "w") as f:
        [f.write(str(elem)+"\n") for elem in walls]