import os
import torch
import numpy as np
from tqdm import tqdm
from scipy.spatial import distance 

from model import MaskRCNN
from utils import load_checkpoint
import config

def track_cells(inference_detections_dir, detection_score_threshold=0.7, max_distance_threshold=50.0):
    tracked_cells_per_frame = {}
    trajectories = {}
    
    detection_files = sorted([f for f in os.listdir(inference_detections_dir) if f.endswith('.pth')])
    
    if not detection_files:
        print("No detection files found. Please run inference first.")
        return tracked_cells_per_frame, trajectories

    print("\nStarting cell tracking...")
    
    next_tracking_id = 1
    
    first_frame_detections = torch.load(os.path.join(inference_detections_dir, detection_files[0]))
    current_frame_cells = []
    
    for i in range(len(first_frame_detections['boxes'])):
        score = first_frame_detections['scores'][i].item()
        if score > detection_score_threshold:
            mask = (first_frame_detections['masks'][i].squeeze(0) > 0.5).cpu().numpy()
            bbox = first_frame_detections['boxes'][i].cpu().numpy()
            
            y_coords, x_coords = np.where(mask)
            centroid = (np.mean(x_coords), np.mean(y_coords))
            
            cell = {
                'id': next_tracking_id,
                'mask': mask,
                'bbox': bbox,
                'centroid': centroid
            }
            current_frame_cells.append(cell)
            trajectories[next_tracking_id] = [centroid]
            next_tracking_id += 1
            
    tracked_cells_per_frame[0] = current_frame_cells
    
    for frame_num in tqdm(range(1, len(detection_files)), desc="Tracking Frames"):
        prev_frame_cells = tracked_cells_per_frame[frame_num - 1]
        
        current_frame_detections = torch.load(os.path.join(inference_detections_dir, detection_files[frame_num]))
        current_frame_cells_new = []
        
        new_cells_untracked = []
        for i in range(len(current_frame_detections['boxes'])):
            score = current_frame_detections['scores'][i].item()
            if score > detection_score_threshold:
                mask = (current_frame_detections['masks'][i].squeeze(0) > 0.5).cpu().numpy()
                bbox = current_frame_detections['boxes'][i].cpu().numpy()
                y_coords, x_coords = np.where(mask)
                centroid = (np.mean(x_coords), np.mean(y_coords))
                new_cells_untracked.append({
                    'mask': mask,
                    'bbox': bbox,
                    'centroid': centroid
                })
        
        matched_new_cells = [False] * len(new_cells_untracked)
        
        for prev_cell in prev_frame_cells:
            prev_centroid = prev_cell['centroid']
            best_match_idx = -1
            min_dist = float('inf')
            
            for new_cell_idx, new_cell in enumerate(new_cells_untracked):
                if not matched_new_cells[new_cell_idx]:
                    new_centroid = new_cell['centroid']
                    dist = distance.euclidean(prev_centroid, new_centroid)
                    
                    if dist < min_dist:
                        min_dist = dist
                        best_match_idx = new_cell_idx
            
            if best_match_idx != -1 and min_dist <= max_distance_threshold:
                matched_new_cells[best_match_idx] = True
                
                new_cell = new_cells_untracked[best_match_idx]
                new_cell['id'] = prev_cell['id']
                current_frame_cells_new.append(new_cell)
                trajectories[prev_cell['id']].append(new_cell['centroid'])
            else:
                pass 
        
        for new_cell_idx, new_cell in enumerate(new_cells_untracked):
            if not matched_new_cells[new_cell_idx]:
                new_cell['id'] = next_tracking_id
                current_frame_cells_new.append(new_cell)
                trajectories[next_tracking_id] = [new_cell['centroid']]
                next_tracking_id += 1
        
        tracked_cells_per_frame[frame_num] = current_frame_cells_new

    print("\nCell tracking complete.")
    
    return tracked_cells_per_frame, trajectories

import numpy as np
from scipy.spatial.distance import euclidean

def analyze_cell_behavior(trajectories):
    cell_metrics = {}

    for cell_id, path in trajectories.items():
        if len(path) < 2:
            cell_metrics[cell_id] = {
                'migration_speed': 0.0,
                'total_displacement': 0.0,
                'total_path_length': 0.0,
                'directionality_ratio': 0.0
            }
            continue

        path = np.array(path)
        
        path_segments = np.diff(path, axis=0)
        segment_lengths = np.linalg.norm(path_segments, axis=1)
        total_path_length = np.sum(segment_lengths)

        total_displacement = euclidean(path[0], path[-1])
        
        migration_speed = np.mean(segment_lengths)
        

        cell_metrics[cell_id] = {
            'migration_speed': migration_speed,
            'total_displacement': total_displacement,
            'total_path_length': total_path_length,
        }

    return cell_metrics

if __name__ == "__main__":
    inference_output_dir = r'dataset\test\01_detections'
    
    tracked_cells_data, trajectories_data = track_cells(
        inference_detections_dir=inference_output_dir,
        detection_score_threshold=0.7,
        max_distance_threshold=50.0
    )
    
    print("\nAnalyzing cell behavior...")
    cell_behavior_metrics = analyze_cell_behavior(trajectories_data)

    torch.save(cell_behavior_metrics, r'dataset\test\01_data\cell_behavior_metrics.pth', weights_only=False)
    torch.save(tracked_cells_data, r'dataset\test\01_data\tracked_cells_data.pth', weights_only=False)
    torch.save(trajectories_data, r'dataset\test\01_data\trajectories_data.pth', weights_only=False)

    print("\n--- Behavioral Analysis Summary ---")
    for cell_id, metrics in cell_behavior_metrics.items():
        print(f"Cell ID {cell_id}:")
        print(f"  - Avg. Speed: {metrics['migration_speed']:.2f} px/frame")
        print(f"  - Total Path: {metrics['total_path_length']:.2f} px")
        print(f"  - Total Displacement: {metrics['total_displacement']:.2f} px")