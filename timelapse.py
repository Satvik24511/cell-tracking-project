import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.animation as animation
import os
import glob
import numpy as np
import config

def visualize_timelapse_animation(image_dir, frame_interval=100):
    image_files = sorted(glob.glob(os.path.join(image_dir, "*.tif")))
    
    if not image_files:
        print(f"No .tif images found in {image_dir}. Please check the path and file type.")
        return

    images = []
    for file_path in image_files:
        image = mpimg.imread(file_path)        
        images.append(image)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(images[0], cmap='gray')
    ax.axis('off')

    def update(frame_number):
        im.set_data(images[frame_number])
        ax.set_title(f"Frame {frame_number}")
        return [im] 

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(images),
        interval=frame_interval,
        blit=False
    )
    plt.show()

if __name__ == "__main__":
    image_folder = os.path.join(config.TEST_DIR, "02")
    visualize_timelapse_animation(image_folder, frame_interval=150)