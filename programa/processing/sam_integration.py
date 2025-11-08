import numpy as np

from segment_anything import sam_model_registry, SamPredictor

def sam_predictor(image, points, model):

    """ SAM 1 """
    # points = np.array(points, dtype=np.float32)
    
    # # SAM predictor
    # sam = sam_model_registry[model](checkpoint=r".\Models\sam_vit_h_4b8939.pth")
    # predictor = SamPredictor(sam)
    # predictor.set_image(image)

    # labels = np.ones(len(points))
    # masks, scores,_ = predictor.predict(
    #     point_coords=points,
    #     point_labels=labels,
    #     multimask_output=True,
    # )

    # best_mask_idx = int(np.argmax(scores))
    # best_mask = masks[best_mask_idx]
    # best_mask_uint8 = (best_mask * 255).astype(np.uint8)

    # return best_mask_uint8

    """SAM 2.1"""

    import cv2
    import numpy as np
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor


    model_map = {
        "base_plus": ("sam2.1_hiera_base_plus.pt", "sam2.1_hiera_b+.yaml"),  # ID 2
        "tiny":  ("sam2.1_hiera_tiny.pt",      "sam2.1_hiera_t.yaml"),
        "small": ("sam2.1_hiera_small.pt",     "sam2.1_hiera_s.yaml"),
        "large": ("sam2.1_hiera_large.pt",     "sam2.1_hiera_l.yaml"),
    }

    if model not in model_map:
        raise ValueError(f"Model {model} not supported. Use: {list(model_map.keys())}")

    ckpt_name, cfg_name = model_map[model]
    ckpt = f"../sam2/checkpoints/{ckpt_name}"
    cfg  = f"../sam2/configs/sam2.1/{cfg_name}"
    

    print("Loading SAM 2.1...")
    sam2_model = build_sam2(cfg, ckpt, device="cpu")
    predictor = SAM2ImagePredictor(sam2_model)

    predictor.set_image(image)
    labels = np.ones(len(points))

    masks, scores, _ = predictor.predict(
        point_coords=points,
        point_labels=labels,
        multimask_output=False  # Single best mask
    )

    best_mask = masks[np.argmax(scores)]
    best_mask_uint8 = (best_mask * 255).astype(np.uint8)

    return best_mask_uint8