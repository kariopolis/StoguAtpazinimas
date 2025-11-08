import cv2
import matplotlib.pyplot as plt

from .file_utils import get_existing_img

from config import margin

def see_img(img, name=None, mask=True):
    
    if mask:
        plt.imshow(img, cmap='gray')
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
    plt.title(name)  # Optional: Show file name as title
    plt.show()

def visualize_points(image, points):
    img = image.copy()
    for p in points:
            cv2.circle(img, (int(p[0]), int(p[1])), 4, (0, 0, 255), -1)  # inner points in red
    plt.imshow(img[..., ::-1])
    plt.title(f"20 Points ({margin*100}% Inward)")
    plt.axis("off")
    plt.show()