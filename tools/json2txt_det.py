import json
import os
from collections import defaultdict

def load_json_files(detections_path, images_path):
    """Load and parse both JSON files."""
    with open(detections_path, 'r') as f:
        detections = json.load(f)
    
    with open(images_path, 'r') as f:
        images_data = json.load(f)
    
    return detections, images_data

def group_by_video(detections, images_data):
    """Group detections by video path."""
    # Create mapping of image_id to file_name
    id_to_path = {img['id']: os.path.dirname(img['file_name']) 
                  for img in images_data['images']}
    
    # Group detections by video path
    video_detections = defaultdict(list)
    for det in detections:
        image_id = det['image_id']
        if image_id in id_to_path:
            video_path = id_to_path[image_id]
            # Convert bbox format from [x,y,w,h] to match required format
            bbox = det['bbox']
            frame_id = next((img['frame_id'] for img in images_data['images'] 
                           if img['id'] == image_id), 0)
            
            # Format: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
            detection_line = [
                frame_id,              # frame
                -1,                    # id (-1 for detection)
                bbox[0],              # bb_left
                bbox[1],              # bb_top
                bbox[2],              # bb_width
                bbox[3],              # bb_height
                det['score'],         # conf
                -1,                   # x (not used in detection)
                -1,                   # y (not used in detection)
                -1                    # z (not used in detection)
            ]
            video_detections[video_path].append(detection_line)
    
    return video_detections

def write_detection_files(video_detections, output_dir):
    """Write detection files for each video."""
    os.makedirs(output_dir, exist_ok=True)
    
    for video_path, detections in video_detections.items():
        # Sort detections by frame number
        detections.sort(key=lambda x: x[0])
        
        # Create output filename
        video_name = os.path.basename(video_path)
        output_file = os.path.join(output_dir, f"{video_name}_det.txt")
        
        # Write detections to file
        with open(output_file, 'w') as f:
            for det in detections:
                line = ','.join(map(str, det))
                f.write(line + '\n')
        
        print(f"Created detection file: {output_file}")

def main():
    # Specify input and output paths
    detections_path = "/home/jiaruili/Documents/github/ByteTrack/YOLOX_para_atk.json"  # Path to your first JSON file
    images_path = "/home/jiaruili/Documents/exp/advTraj/baselines/parallel_baseline_atk/annotations/test.json"          # Path to your second JSON file
    output_dir = "/home/jiaruili/Documents/exp/advTraj/baselines/parallel_baseline_atk/dets"     # Output directory for det.txt files
    
    # Process the files
    detections, images_data = load_json_files(detections_path, images_path)
    video_detections = group_by_video(detections, images_data)
    write_detection_files(video_detections, output_dir)

if __name__ == "__main__":
    main()
