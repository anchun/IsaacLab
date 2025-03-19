import os
import numpy as np
import matplotlib.pyplot as plt

image_folder = './camera/'
depth_images = [img for img in os.listdir(image_folder) if img.endswith(".npy")]
depth_images.sort()
for depth_image in depth_images:
    depth_image_path = os.path.join(image_folder, depth_image)
    depth_map = np.load(depth_image_path).squeeze()
    depth_min, depth_max = depth_map.min(), depth_map.max()
    depth_normalized = (depth_map - depth_min) / (depth_max - depth_min + 1e-8)
    out_image_path = os.path.splitext(depth_image_path)[0] + ".png"
    print("converting", depth_image_path)
    plt.imsave(out_image_path, depth_normalized, cmap="viridis")