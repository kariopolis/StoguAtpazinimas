import numpy as np
import cv2
import matplotlib.pyplot as plt


def extract_roof_structure_lines(ortho_rgb, roof_mask, debug=True):
    gray = cv2.cvtColor(ortho_rgb, cv2.COLOR_RGB2GRAY)

    # 1. Smooth strongly (roof-scale)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    # 2. Sobel gradients
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)

    # 3. Normalize
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 4. Adaptive threshold
    edges = cv2.adaptiveThreshold(
        mag,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        -5
    )

    # 5. Restrict to roof
    edges = cv2.bitwise_and(edges, roof_mask)
    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        iterations=1
    )

    # Hough line detection (LONG lines only)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=60,
        maxLineGap=10
    )

    if lines is None:
        return [], edges

    # Filter lines by length
    structure_lines = []
    for l in lines:
        x1, y1, x2, y2 = l[0]
        length = np.hypot(x2 - x1, y2 - y1)
        if length > 60:
            structure_lines.append((x1, y1, x2, y2))

    if debug:
        dbg = ortho_rgb.copy()
        for x1, y1, x2, y2 in structure_lines:
            cv2.line(dbg, (x1, y1), (x2, y2), (0, 255, 0), 2)

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(roof_mask, cmap="gray")
        plt.title("Roof mask"); plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(edges, cmap="gray")
        plt.title("Edges inside roof"); plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(dbg)
        plt.title("Structural roof lines"); plt.axis("off")

        plt.tight_layout()
        plt.show()

    return structure_lines, edges




def extract_planes(structural_lines, edges, roof_mask, ortho_img):
    separator = np.zeros_like(roof_mask)

    for line in structural_lines:  # lines from Hough
        x1, y1, x2, y2 = line
        cv2.line(separator, (x1, y1), (x2, y2), 255, 3)
    
    cut_mask = roof_mask.copy()
    cut_mask[separator > 0] = 0

    num_labels, labels = cv2.connectedComponents(cut_mask)

    planes = []
    for i in range(1, num_labels):
        plane = np.zeros_like(roof_mask)
        plane[labels == i] = 255
        if cv2.countNonZero(plane) > 500:  # area threshold
            planes.append(plane)

    plane_vis = ortho_img.copy()

    colors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0)]
    for i, p in enumerate(planes):
        plane_vis[p > 0] = (
            0.5 * plane_vis[p > 0] +
            0.5 * np.array(colors[i % len(colors)])
        )

    plt.imshow(plane_vis)
    plt.title("Roof planes")
    plt.axis("off")
    plt.show()






ortho_img=cv2.imread(r"C:\Users\karov\Documents\Robotika\StoguAtpazinimas\Data\Modified\Ortho\Img_4.png")
only_roof_img=cv2.imread(r"C:\Users\karov\Documents\Robotika\StoguAtpazinimas\Data\OnlyRoof\roof_Img_4.png")
gray = cv2.cvtColor(only_roof_img, cv2.COLOR_RGB2GRAY)
_, roof_mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
roof_mask = roof_mask.astype(np.uint8)
structural_lines, edges = extract_roof_structure_lines(ortho_img,roof_mask)
extract_planes(structural_lines, edges, roof_mask, ortho_img)
