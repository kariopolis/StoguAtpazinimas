from data_utils import get_roof_only_img, create_mask, get_existing_mask, see_img, get_existing_img, visualize_points, crop_image_according_to_mask, save_roof_img, save_wall_measurements
from processing import finding_closest_contour, generate_points_inside_contour, scale_contour, sam_predictor, mask_adjustment, roof_elements_detection
from processing.proc_utils import calculate_wall_measurements

from config import data_creation, scale_factor, segment_roof, extract_obstacles

if data_creation: mask, image, mask_id = create_mask(data_img_id = 1)
elif segment_roof==1:
    mask_id=int(input("Input Mask ID which you would like to use. \n"))
    print(f"Getting mask number {mask_id+1}")
    mask = get_existing_mask(mask_id)
    image = get_existing_img(mask_id)
elif extract_obstacles:
    roof_img_id=int(input("Input roof only image ID which you would like to use. \n"))
    print(f"Getting image number {roof_img_id+1}")
    roof_image = get_roof_only_img(roof_img_id)
    see_img(roof_image, f"Roof image num {roof_img_id+1}",mask=False)

if segment_roof==1:
    see_img(mask)
    #Leave only the clostest contour to the center of the mask
    mask_edited, closest_contour = finding_closest_contour(mask)

    #Generate points inside the polygon mask
    points_for_sam=generate_points_inside_contour(closest_contour)

    visualize_points(image, points_for_sam)

    #Scale mask and keep only corner points
    scaled_mask, scaled_contour = scale_contour(closest_contour, scale_factor, mask_edited.shape)

    #Four models to choose from: tiny/small/base_plus/large
    predicted_mask = sam_predictor(image, points_for_sam, model="large")
    see_img(predicted_mask)

    #Mask shape adjustment according to the house shape taken from polygon picture
    adjusted_mask, adjusted_contour =mask_adjustment(predicted_mask, scaled_mask, scaled_contour, image, visualize=True)

    #Crop the roof out of image accoring to the mask
    cropped_roof_image= crop_image_according_to_mask(adjusted_contour,image)
    see_img(cropped_roof_image, mask=False)

    mask_id = mask_id-1 if data_creation else mask_id

    save_roof_img(cropped_roof_image, mask_id)

    walls=calculate_wall_measurements(adjusted_contour)

    save_wall_measurements(walls, mask_id)

roof_elements_detection(roof_image)






