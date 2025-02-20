import os
import tempfile
import subprocess
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import numpy as np
from tqdm import tqdm
from persistence import load_detection_data

def create_frame_data(json_path):
    """Create frame-by-frame detection data for visualization."""
    try:
        data = load_detection_data(json_path)
        if not data:
            print("No data loaded from JSON file")
            return None
        
        if "video_metadata" not in data or "frame_detections" not in data:
            print("Invalid JSON structure: missing required fields")
            return None
        
        # Extract video metadata
        metadata = data["video_metadata"]
        if "fps" not in metadata or "total_frames" not in metadata:
            print("Invalid metadata: missing fps or total_frames")
            return None
            
        fps = metadata["fps"]
        total_frames = metadata["total_frames"]
        
        # Create frame data
        frame_counts = {}
        for frame_data in data["frame_detections"]:
            if "frame" not in frame_data or "objects" not in frame_data:
                continue  # Skip invalid frame data
            frame_num = frame_data["frame"]
            frame_counts[frame_num] = len(frame_data["objects"])
        
        # Fill in missing frames with 0 detections
        for frame in range(total_frames):
            if frame not in frame_counts:
                frame_counts[frame] = 0
        
        if not frame_counts:
            print("No valid frame data found")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(list(frame_counts.items()), columns=["frame", "detections"])
        df["timestamp"] = df["frame"] / fps
        
        return df, metadata
        
    except Exception as e:
        print(f"Error creating frame data: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def generate_frame_image(df, frame_num, temp_dir, max_y):
    """Generate and save a single frame of the visualization."""
    # Set the style to dark background
    plt.style.use('dark_background')
    
    # Set global font to monospace
    plt.rcParams['font.family'] = 'monospace'
    plt.rcParams['font.monospace'] = ['DejaVu Sans Mono']
    
    plt.figure(figsize=(10, 6))
    
    # Plot data up to current frame
    current_data = df[df['frame'] <= frame_num]
    plt.plot(df['frame'], df['detections'], color='#1a1a1a', alpha=0.5)  # Darker background line
    plt.plot(current_data['frame'], current_data['detections'], color='#00ff41')  # Matrix green
    
    # Add vertical line for current position
    plt.axvline(x=frame_num, color='#ff0000', linestyle='-', alpha=0.7)  # Keep red for position
    
    # Set consistent axes
    plt.xlim(0, len(df) - 1)
    plt.ylim(0, max_y * 1.1)  # Add 10% padding
    
    # Add labels with Matrix green color
    plt.title(f'FRAME {frame_num:04d} - DETECTIONS OVER TIME', color='#00ff41', pad=20)
    plt.xlabel('FRAME NUMBER', color='#00ff41')
    plt.ylabel('NUMBER OF DETECTIONS', color='#00ff41')
    
    # Add current stats in Matrix green with monospace formatting
    current_detections = df[df['frame'] == frame_num]['detections'].iloc[0]
    plt.text(0.02, 0.98, f'CURRENT DETECTIONS: {current_detections:02d}', 
             transform=plt.gca().transAxes, verticalalignment='top',
             color='#00ff41', family='monospace')
    
    # Style the grid and ticks
    plt.grid(True, color='#1a1a1a', linestyle='-', alpha=0.3)
    plt.tick_params(colors='#00ff41')
    
    # Save frame
    frame_path = os.path.join(temp_dir, f'frame_{frame_num:05d}.png')
    plt.savefig(frame_path, bbox_inches='tight', dpi=100, facecolor='black', edgecolor='none')
    plt.close()
    
    return frame_path

def generate_gauge_frame(df, frame_num, temp_dir, detect_keyword="OBJECT"):
    """Generate a modern square-style binary gauge visualization frame."""
    # Set the style to dark background
    plt.style.use('dark_background')
    
    # Set global font to monospace
    plt.rcParams['font.family'] = 'monospace'
    plt.rcParams['font.monospace'] = ['DejaVu Sans Mono']
    
    # Create figure with 16:9 aspect ratio
    plt.figure(figsize=(16, 9))
    
    # Get current detection state
    current_detections = df[df['frame'] == frame_num]['detections'].iloc[0]
    has_detection = current_detections > 0
    
    # Create a simple gauge visualization
    plt.axis('off')
    
    # Set colors
    if has_detection:
        color = '#00ff41'  # Matrix green for YES
        status = 'YES'
        indicator_pos = 0.8  # Right position
    else:
        color = '#ff0000'  # Red for NO
        status = 'NO'
        indicator_pos = 0.2  # Left position
    
    # Draw background rectangle
    background = plt.Rectangle((0.1, 0.3), 0.8, 0.2, 
                             facecolor='#1a1a1a', 
                             edgecolor='#333333',
                             linewidth=2)
    plt.gca().add_patch(background)
    
    # Draw indicator
    indicator_width = 0.05
    indicator = plt.Rectangle((indicator_pos - indicator_width/2, 0.25), 
                            indicator_width, 0.3,
                            facecolor=color,
                            edgecolor=None)
    plt.gca().add_patch(indicator)
    
    # Add tick marks
    tick_positions = [0.2, 0.5, 0.8]  # NO, CENTER, YES
    for x in tick_positions:
        plt.plot([x, x], [0.3, 0.5], color='#444444', linewidth=2)
    
    # Add YES/NO labels
    plt.text(0.8, 0.2, 'YES', color='#00ff41', fontsize=14,
             ha='center', va='center', family='monospace')
    plt.text(0.2, 0.2, 'NO', color='#ff0000', fontsize=14,
             ha='center', va='center', family='monospace')
    
    # Add status box at top with detection keyword
    plt.text(0.5, 0.8, f'{detect_keyword.upper()} DETECTED?', color=color,
             fontsize=16, ha='center', va='center', family='monospace',
             bbox=dict(facecolor='#1a1a1a', 
                      edgecolor=color,
                      linewidth=2,
                      pad=10))
    
    # Add frame counter at bottom
    plt.text(0.5, 0.1, f'FRAME: {frame_num:04d}', color='#00ff41',
             fontsize=14, ha='center', va='center', family='monospace')
    
    # Add subtle grid lines for depth
    for x in np.linspace(0.2, 0.8, 7):
        plt.plot([x, x], [0.3, 0.5], color='#222222', linewidth=1, zorder=0)
    
    # Add glow effect to indicator
    for i in range(3):
        glow = plt.Rectangle((indicator_pos - (indicator_width + i*0.01)/2, 
                            0.25 - i*0.01),
                            indicator_width + i*0.01, 
                            0.3 + i*0.02,
                            facecolor=color,
                            alpha=0.1/(i+1))
        plt.gca().add_patch(glow)
    
    # Set consistent plot limits
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    # Save frame with 16:9 aspect ratio
    frame_path = os.path.join(temp_dir, f'gauge_{frame_num:05d}.png')
    plt.savefig(frame_path, 
                bbox_inches='tight', 
                dpi=100, 
                facecolor='black', 
                edgecolor='none',
                pad_inches=0)
    plt.close()
    
    return frame_path

def create_video_visualization(json_path, style="timeline"):
    """Create a video visualization of the detection data."""
    try:
        if not json_path:
            return None, "No JSON file provided"
            
        if not os.path.exists(json_path):
            return None, f"File not found: {json_path}"
            
        # Load and process data
        result = create_frame_data(json_path)
        if result is None:
            return None, "Failed to load detection data from JSON file"
            
        frame_data, metadata = result
        if len(frame_data) == 0:
            return None, "No frame data found in JSON file"
        
        total_frames = metadata["total_frames"]
        detect_keyword = metadata.get("detect_keyword", "OBJECT")  # Get the detection keyword
        
        # Create temporary directory for frames
        with tempfile.TemporaryDirectory() as temp_dir:
            max_y = frame_data['detections'].max()
            
            # Generate each frame
            print("Generating frames...")
            frame_paths = []
            with tqdm(total=total_frames, desc="Generating frames") as pbar:
                for frame in range(total_frames):
                    try:
                        if style == "gauge":
                            frame_path = generate_gauge_frame(frame_data, frame, temp_dir, detect_keyword)
                        else:  # default to timeline
                            frame_path = generate_frame_image(frame_data, frame, temp_dir, max_y)
                        if frame_path and os.path.exists(frame_path):
                            frame_paths.append(frame_path)
                        else:
                            print(f"Warning: Failed to generate frame {frame}")
                        pbar.update(1)
                    except Exception as e:
                        print(f"Error generating frame {frame}: {str(e)}")
                        continue
            
            if not frame_paths:
                return None, "Failed to generate any frames"
                
            # Create output video path
            output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
            os.makedirs(output_dir, exist_ok=True)
            output_video = os.path.join(output_dir, f"detection_visualization_{style}.mp4")
            
            # Create temp output path
            base, ext = os.path.splitext(output_video)
            temp_output = f"{base}_temp{ext}"
            
            # First pass: Create video with OpenCV VideoWriter
            print("Creating initial video...")
            # Get frame size from first image
            first_frame = cv2.imread(frame_paths[0])
            height, width = first_frame.shape[:2]
            
            out = cv2.VideoWriter(
                temp_output,
                cv2.VideoWriter_fourcc(*"mp4v"),
                metadata["fps"],
                (width, height)
            )
            
            with tqdm(total=total_frames, desc="Creating video") as pbar:  # Use total_frames here too
                for frame_path in frame_paths:
                    frame = cv2.imread(frame_path)
                    out.write(frame)
                    pbar.update(1)
            
            out.release()
            
            # Second pass: Convert to web-compatible format
            print("Converting to web format...")
            try:
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-i",
                        temp_output,
                        "-c:v",
                        "libx264",
                        "-preset",
                        "medium",
                        "-crf",
                        "23",
                        "-movflags",
                        "+faststart",  # Better web playback
                        "-loglevel",
                        "error",
                        output_video,
                    ],
                    check=True,
                )

                os.remove(temp_output)  # Remove the temporary file

                if not os.path.exists(output_video):
                    print(f"Warning: FFmpeg completed but output file not found at {output_video}")
                    return None, "Failed to create video"

                # Return video path and stats
                stats = f"""Video Stats:
FPS: {metadata['fps']}
Total Frames: {metadata['total_frames']}
Duration: {metadata['duration_sec']:.2f} seconds
Max Detections in a Frame: {frame_data['detections'].max()}
Average Detections per Frame: {frame_data['detections'].mean():.2f}"""
                
                return output_video, stats

            except subprocess.CalledProcessError as e:
                print(f"Error running FFmpeg: {str(e)}")
                if os.path.exists(temp_output):
                    os.remove(temp_output)
                return None, f"Error creating visualization: {str(e)}"
        
    except Exception as e:
        print(f"Error creating video visualization: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, f"Error creating visualization: {str(e)}" 