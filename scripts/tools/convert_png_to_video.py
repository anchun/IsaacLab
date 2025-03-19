import cv2
import os

def images_to_video2(image_folder, video_name, prefix, fps=30):
    images = [img for img in os.listdir(image_folder) if img.endswith(".png") and img.startswith(prefix)]
    images.sort()

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

    print("writing: ", video_name)
    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        video.write(frame)
    print("writing done. ", video_name)

    video.release()


image_folder = './camera/' # input image folder
images_to_video2(image_folder, './output_rgb.mp4', "rgb_")
images_to_video2(image_folder, './output_depth.mp4', "distance_")
images_to_video2(image_folder, './output_normal.mp4', "normals_")
images_to_video2(image_folder, './output_segmentation.mp4', "semantic_")