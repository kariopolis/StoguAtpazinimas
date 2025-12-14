
#Main portion of the path for taking and saving data images
main_data_path = r"C:\Users\karov\Documents\Robotika\StoguAtpazinimas\Data"

#Modified data dimensions (512x512 pix)
working_img_size = 512

#To create new modified data the value should be 1
data_creation=0

#Resolution of the pixel in the data image. 1pix=13cm in real world
pixel_size=0.131

#Number of points to generate inside the roof poligon for better SAM detection
reference_points_num=10

#Spacing between poligon edge and points (10%)
margin = 0.1

#Scale factor
scale_factor=1.2

#Angle threshold for house corners detection
angle_threshold=150#

#Flag for modified data segmentation to roof only img. Value should be 1 if function required
segment_roof=0

#Flag for modified data obstacles extraction from roof only img. Value should be 1 if function required
extract_obstacles=1

