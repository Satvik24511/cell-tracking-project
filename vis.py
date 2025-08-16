import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import imageio

def get_unique_colors(n):
    cmap = plt.cm.get_cmap('hsv')
    return [cmap(i) for i in np.linspace(0, 1, n)]

def create_tracking_video(image_dir, tracked_cells_per_frame, trajectories, cell_behavior_metrics, target_cell_id, output_path, fps=10):
    
    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith('.tif')])

    if not image_files:
        print("No images found for visualization.")
        return

    all_cell_ids = list(trajectories.keys())
    if not all_cell_ids:
        print("No cell trajectories found. Cannot create video.")
        return
    
    print(f"\nTracking and visualizing a single cell: ID {target_cell_id}")
    output_path = f"cell_tracking_animation_{target_cell_id}.gif"

    id_to_color_map = { cell_id: get_unique_colors(len(all_cell_ids))[i] for i, cell_id in enumerate(all_cell_ids) }

    print("\nStarting video creation...")

    first_image = cv2.imread(os.path.join(image_dir, image_files[0]))
    frame_h, frame_w, _ = first_image.shape
    
    text_height = 120
    
    canvas_w = frame_w * 2
    canvas_h = frame_h + text_height
    
    black_color = (0, 0, 0)
    
    start_frame = -1
    for i in range(len(image_files)):
        cells = tracked_cells_per_frame.get(i, [])
        for cell in cells:
            if cell['id'] == target_cell_id:
                start_frame = i
                break
        if start_frame != -1:
            break

    with imageio.get_writer(output_path, mode='I', fps=fps) as writer:
        for frame_num in tqdm(range(len(image_files)), desc="Rendering Frames"):
            img_path = os.path.join(image_dir, image_files[frame_num])
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            left_panel = cv2.resize(image, (frame_w, frame_h))
            cv2.putText(left_panel, "Original Timelapse", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
            right_panel = cv2.resize(image.copy(), (frame_w, frame_h))
            cv2.putText(right_panel, "Tracking Overlay", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            trajectory_points = trajectories.get(target_cell_id, [])
            path_to_draw = trajectory_points[max(0, start_frame):frame_num + 1]

            if len(path_to_draw) > 1:
                color_bgr = tuple(int(c * 255) for c in id_to_color_map[target_cell_id][:3])
                path_to_draw_np = np.array(path_to_draw, dtype=np.int32)
                cv2.polylines(right_panel, [path_to_draw_np], False, color_bgr, 2)
            
            target_cell = None
            for cell in tracked_cells_per_frame.get(frame_num, []):
                if cell['id'] == target_cell_id:
                    target_cell = cell
                    break
            
            if target_cell:
                mask = target_cell['mask']
                mask_indices = np.where(mask > 0)
                
                mask_color = (0, 255, 0)
                mask_overlay = np.zeros_like(right_panel, dtype=np.uint8)
                mask_overlay[mask_indices] = mask_color
                
                right_panel = cv2.addWeighted(right_panel, 0.7, mask_overlay, 0.3, 0)
                
                bbox = target_cell['bbox']
                cv2.rectangle(right_panel, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255,255,255), 2)
                cv2.rectangle(left_panel, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0,255,0), 2)

            metrics_panel = np.zeros((text_height, canvas_w, 3), dtype=np.uint8)
            metrics_panel[:, :] = black_color
            
            metrics = cell_behavior_metrics.get(target_cell_id, {})
            speed = metrics.get('migration_speed', 0.0)
            displacement = metrics.get('total_displacement', 0.0)
            path_length = metrics.get('total_path_length', 0.0)
            
            text_stats = [
                f"Tracking Cell ID: {target_cell_id}",
                f"Avg. Speed: {speed:.2f} px/frame",
                f"Total Path Length: {path_length:.2f} px",
                f"Total Displacement: {displacement:.2f} px",
                f"Frame: {frame_num} / {len(image_files) - 1}"
            ]
            
            for i, text in enumerate(text_stats):
                cv2.putText(metrics_panel, text, (10, 20 + i * 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            combined_frame = np.concatenate((left_panel, right_panel), axis=1)
            cv2.line(combined_frame, (frame_w, 0), (frame_w, frame_h), (255, 255, 255), 2)
            
            final_image = np.concatenate((combined_frame, metrics_panel), axis=0)
            writer.append_data(final_image)

    print(f"\nVideo saved to {output_path}")

def main():
    tracked_cells_data = torch.load(r'dataset\test\01_data\tracked_cells_data.pth', weights_only=False)
    trajectories_data = torch.load(r'dataset\test\01_data\trajectories_data.pth', weights_only=False)
    cell_behavior_metrics = torch.load(r'dataset\test\01_data\cell_behavior_metrics.pth', weights_only=False)

    image_dir = r'dataset\test\01'
    
    all_cells_sorted = sorted(cell_behavior_metrics.items(), key=lambda item: item[1]['total_path_length'], reverse=True)
    
    top_10_cell_ids = [cell_id for cell_id, metrics in all_cells_sorted[:10]]
    
    for cell_id in top_10_cell_ids:
        print(f"Creating video for Cell ID: {cell_id}")
        create_tracking_video(image_dir, tracked_cells_data, trajectories_data, cell_behavior_metrics, cell_id, output_path=f"cell_tracking_animation_{cell_id}.gif", fps=10)
        
if __name__ == "__main__":
    main()
