import pandas as pd
import matplotlib.pyplot as plt
from persistence import load_detection_data
import argparse

def visualize_detections(json_path):
    """
    Visualize detection data from a JSON file.
    
    Args:
        json_path (str): Path to the JSON file containing detection data.
    """
    # Load the persisted JSON data
    data = load_detection_data(json_path)
    if not data:
        return

    # Convert the frame detections to a DataFrame
    rows = []
    for frame_data in data["frame_detections"]:
        frame = frame_data["frame"]
        timestamp = frame_data["timestamp"]
        for obj in frame_data["objects"]:
            rows.append({
                "frame": frame,
                "timestamp": timestamp,
                "keyword": obj["keyword"],
                "x1": obj["bbox"][0],
                "y1": obj["bbox"][1],
                "x2": obj["bbox"][2],
                "y2": obj["bbox"][3],
                "area": (obj["bbox"][2] - obj["bbox"][0]) * (obj["bbox"][3] - obj["bbox"][1])
            })

    if not rows:
        print("No detections found in the data")
        return

    df = pd.DataFrame(rows)

    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(15, 10))
    
    # Plot 1: Number of detections per frame
    plt.subplot(2, 2, 1)
    detections_per_frame = df.groupby("frame").size()
    plt.plot(detections_per_frame.index, detections_per_frame.values)
    plt.xlabel("Frame")
    plt.ylabel("Number of Detections")
    plt.title("Detections Per Frame")

    # Plot 2: Distribution of detection areas
    plt.subplot(2, 2, 2)
    df["area"].hist(bins=30)
    plt.xlabel("Detection Area (normalized)")
    plt.ylabel("Count")
    plt.title("Distribution of Detection Areas")

    # Plot 3: Average detection area over time
    plt.subplot(2, 2, 3)
    avg_area = df.groupby("frame")["area"].mean()
    plt.plot(avg_area.index, avg_area.values)
    plt.xlabel("Frame")
    plt.ylabel("Average Detection Area")
    plt.title("Average Detection Area Over Time")

    # Plot 4: Heatmap of detection centers
    plt.subplot(2, 2, 4)
    df["center_x"] = (df["x1"] + df["x2"]) / 2
    df["center_y"] = (df["y1"] + df["y2"]) / 2
    plt.hist2d(df["center_x"], df["center_y"], bins=30)
    plt.colorbar()
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("Detection Center Heatmap")

    # Adjust layout and display
    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Total frames analyzed: {len(data['frame_detections'])}")
    print(f"Total detections: {len(df)}")
    print(f"Average detections per frame: {len(df) / len(data['frame_detections']):.2f}")
    print(f"\nVideo metadata:")
    for key, value in data["video_metadata"].items():
        print(f"{key}: {value}")

def main():
    parser = argparse.ArgumentParser(description="Visualize object detection data")
    parser.add_argument("json_file", help="Path to the JSON file containing detection data")
    args = parser.parse_args()
    
    visualize_detections(args.json_file)

if __name__ == "__main__":
    main() 