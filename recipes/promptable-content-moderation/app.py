#!/usr/bin/env python3
import gradio as gr
import os
from main import load_moondream, process_video, load_sam_model
import shutil
import torch
from visualization import visualize_detections
from persistence import load_detection_data
import matplotlib.pyplot as plt
import io
from PIL import Image
import pandas as pd
from video_visualization import create_video_visualization

# Get absolute path to workspace root
WORKSPACE_ROOT = os.path.dirname(os.path.abspath(__file__))

# Check CUDA availability
print(f"Is CUDA available: {torch.cuda.is_available()}")
# We want to get True
print(f"CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
# GPU Name

# Initialize Moondream model globally for reuse (will be loaded on first use)
model, tokenizer = None, None


def process_video_file(
    video_file,
    target_object,
    box_style,
    ffmpeg_preset,
    grid_rows,
    grid_cols,
    test_mode,
    test_duration,
):
    """Process a video file through the Gradio interface."""
    try:
        if not video_file:
            raise gr.Error("Please upload a video file")

        # Load models if not already loaded
        global model, tokenizer
        if model is None or tokenizer is None:
            model, tokenizer = load_moondream()

        # Ensure input/output directories exist using absolute paths
        inputs_dir = os.path.join(WORKSPACE_ROOT, "inputs")
        outputs_dir = os.path.join(WORKSPACE_ROOT, "outputs")
        os.makedirs(inputs_dir, exist_ok=True)
        os.makedirs(outputs_dir, exist_ok=True)

        # Copy uploaded video to inputs directory
        video_filename = f"input_{os.path.basename(video_file)}"
        input_video_path = os.path.join(inputs_dir, video_filename)
        shutil.copy2(video_file, input_video_path)

        try:
            # Process the video
            output_path = process_video(
                input_video_path,
                target_object,
                test_mode=test_mode,
                test_duration=test_duration,
                ffmpeg_preset=ffmpeg_preset,
                grid_rows=grid_rows,
                grid_cols=grid_cols,
                box_style=box_style,
            )

            # Get the corresponding JSON path
            base_name = os.path.splitext(os.path.basename(video_filename))[0]
            json_path = os.path.join(
                outputs_dir, f"{box_style}_{target_object}_{base_name}_detections.json"
            )

            # Verify output exists and is readable
            if not output_path or not os.path.exists(output_path):
                print(f"Warning: Output path {output_path} does not exist")
                # Try to find the output based on expected naming convention
                expected_output = os.path.join(
                    outputs_dir, f"{box_style}_{target_object}_{video_filename}"
                )
                if os.path.exists(expected_output):
                    output_path = expected_output
                else:
                    # Try searching in outputs directory for any matching file
                    matching_files = [
                        f
                        for f in os.listdir(outputs_dir)
                        if f.startswith(f"{box_style}_{target_object}_")
                    ]
                    if matching_files:
                        output_path = os.path.join(outputs_dir, matching_files[0])
                    else:
                        raise gr.Error("Failed to locate output video")

            # Convert output path to absolute path if it isn't already
            if not os.path.isabs(output_path):
                output_path = os.path.join(WORKSPACE_ROOT, output_path)

            print(f"Returning output path: {output_path}")
            return output_path, json_path

        finally:
            # Clean up input file
            try:
                if os.path.exists(input_video_path):
                    os.remove(input_video_path)
            except:
                pass

    except Exception as e:
        print(f"Error in process_video_file: {str(e)}")
        raise gr.Error(f"Error processing video: {str(e)}")


def create_visualization_plots(json_path):
    """Create visualization plots and return them as images."""
    try:
        # Load the data
        data = load_detection_data(json_path)
        if not data:
            return None, None, None, None, None, None, None, None, "No data found"

        # Convert to DataFrame
        rows = []
        for frame_data in data["frame_detections"]:
            frame = frame_data["frame"]
            timestamp = frame_data["timestamp"]
            for obj in frame_data["objects"]:
                rows.append(
                    {
                        "frame": frame,
                        "timestamp": timestamp,
                        "keyword": obj["keyword"],
                        "x1": obj["bbox"][0],
                        "y1": obj["bbox"][1],
                        "x2": obj["bbox"][2],
                        "y2": obj["bbox"][3],
                        "area": (obj["bbox"][2] - obj["bbox"][0])
                        * (obj["bbox"][3] - obj["bbox"][1]),
                        "center_x": (obj["bbox"][0] + obj["bbox"][2]) / 2,
                        "center_y": (obj["bbox"][1] + obj["bbox"][3]) / 2,
                    }
                )

        if not rows:
            return (
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                "No detections found in the data",
            )

        df = pd.DataFrame(rows)
        plots = []

        # Create each plot and convert to image
        for plot_num in range(8):  # Increased to 8 plots
            plt.figure(figsize=(8, 6))

            if plot_num == 0:
                # Plot 1: Number of detections per frame (Original)
                detections_per_frame = df.groupby("frame").size()
                plt.plot(detections_per_frame.index, detections_per_frame.values)
                plt.xlabel("Frame")
                plt.ylabel("Number of Detections")
                plt.title("Detections Per Frame")

            elif plot_num == 1:
                # Plot 2: Distribution of detection areas (Original)
                df["area"].hist(bins=30)
                plt.xlabel("Detection Area (normalized)")
                plt.ylabel("Count")
                plt.title("Distribution of Detection Areas")

            elif plot_num == 2:
                # Plot 3: Average detection area over time (Original)
                avg_area = df.groupby("frame")["area"].mean()
                plt.plot(avg_area.index, avg_area.values)
                plt.xlabel("Frame")
                plt.ylabel("Average Detection Area")
                plt.title("Average Detection Area Over Time")

            elif plot_num == 3:
                # Plot 4: Heatmap of detection centers (Original)
                plt.hist2d(df["center_x"], df["center_y"], bins=30)
                plt.colorbar()
                plt.xlabel("X Position")
                plt.ylabel("Y Position")
                plt.title("Detection Center Heatmap")

            elif plot_num == 4:
                # Plot 5: Time-based Detection Density
                # Shows when in the video most detections occur
                df["time_bucket"] = pd.qcut(df["timestamp"], q=20, labels=False)
                time_density = df.groupby("time_bucket").size()
                plt.bar(time_density.index, time_density.values)
                plt.xlabel("Video Timeline (20 segments)")
                plt.ylabel("Number of Detections")
                plt.title("Detection Density Over Video Duration")

            elif plot_num == 5:
                # Plot 6: Screen Region Analysis
                # Divide screen into 3x3 grid and show detection counts
                try:
                    df["grid_x"] = pd.qcut(
                        df["center_x"],
                        q=3,
                        labels=["Left", "Center", "Right"],
                        duplicates="drop",
                    )
                    df["grid_y"] = pd.qcut(
                        df["center_y"],
                        q=3,
                        labels=["Top", "Middle", "Bottom"],
                        duplicates="drop",
                    )
                    region_counts = (
                        df.groupby(["grid_y", "grid_x"]).size().unstack(fill_value=0)
                    )
                    plt.imshow(region_counts, cmap="YlOrRd")
                    plt.colorbar(label="Detection Count")
                    for i in range(3):
                        for j in range(3):
                            plt.text(
                                j, i, region_counts.iloc[i, j], ha="center", va="center"
                            )
                    plt.xticks(range(3), ["Left", "Center", "Right"])
                    plt.yticks(range(3), ["Top", "Middle", "Bottom"])
                    plt.title("Screen Region Analysis")
                except Exception as e:
                    plt.text(
                        0.5,
                        0.5,
                        "Insufficient variation in detection positions",
                        ha="center",
                        va="center",
                    )
                    plt.title("Screen Region Analysis (Not Available)")

            elif plot_num == 6:
                # Plot 7: Detection Size Categories
                # Categorize detections by size for content moderation
                try:
                    size_labels = [
                        "Small (likely far/background)",
                        "Medium-small",
                        "Medium-large",
                        "Large (likely foreground/close)",
                    ]

                    # Handle cases with limited unique values
                    unique_areas = df["area"].nunique()
                    if unique_areas >= 4:
                        df["size_category"] = pd.qcut(
                            df["area"], q=4, labels=size_labels, duplicates="drop"
                        )
                    else:
                        # Alternative binning for limited unique values
                        df["size_category"] = pd.cut(
                            df["area"],
                            bins=unique_areas,
                            labels=size_labels[:unique_areas],
                        )

                    size_dist = df["size_category"].value_counts()
                    plt.pie(size_dist.values, labels=size_dist.index, autopct="%1.1f%%")
                    plt.title("Detection Size Distribution")
                except Exception as e:
                    plt.text(
                        0.5,
                        0.5,
                        "Insufficient variation in detection sizes",
                        ha="center",
                        va="center",
                    )
                    plt.title("Detection Size Distribution (Not Available)")

            elif plot_num == 7:
                # Plot 8: Temporal Pattern Analysis
                # Show patterns of when detections occur in sequence
                try:
                    detection_gaps = df.sort_values("frame")["frame"].diff()
                    if len(detection_gaps.dropna().unique()) > 1:
                        plt.hist(
                            detection_gaps.dropna(),
                            bins=min(30, len(detection_gaps.dropna().unique())),
                            edgecolor="black",
                        )
                        plt.xlabel("Frames Between Detections")
                        plt.ylabel("Frequency")
                        plt.title("Detection Temporal Pattern Analysis")
                    else:
                        plt.text(
                            0.5,
                            0.5,
                            "Uniform detection intervals",
                            ha="center",
                            va="center",
                        )
                        plt.title("Temporal Pattern Analysis (Uniform)")
                except Exception as e:
                    plt.text(
                        0.5, 0.5, "Insufficient temporal data", ha="center", va="center"
                    )
                    plt.title("Temporal Pattern Analysis (Not Available)")

            # Save plot to bytes
            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            plots.append(Image.open(buf))
            plt.close()

        # Enhanced summary text
        summary = f"""Summary Statistics:
Total frames analyzed: {len(data['frame_detections'])}
Total detections: {len(df)}
Average detections per frame: {len(df) / len(data['frame_detections']):.2f}

Detection Patterns:
- Peak detection count: {df.groupby('frame').size().max()} (in a single frame)
- Most common screen region: {df.groupby(['grid_y', 'grid_x']).size().idxmax()}
- Average detection size: {df['area'].mean():.3f}
- Median frames between detections: {detection_gaps.median():.1f}

Video metadata:
"""
        for key, value in data["video_metadata"].items():
            summary += f"{key}: {value}\n"

        return (
            plots[0],
            plots[1],
            plots[2],
            plots[3],
            plots[4],
            plots[5],
            plots[6],
            plots[7],
            summary,
        )

    except Exception as e:
        print(f"Error creating visualization: {str(e)}")
        import traceback

        traceback.print_exc()
        return (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            f"Error creating visualization: {str(e)}",
        )


# Create the Gradio interface
with gr.Blocks(title="Promptable Content Moderation") as app:
    with gr.Tabs():
        with gr.Tab("Process Video"):
            gr.Markdown("# Promptable Content Moderation with Moondream")
            gr.Markdown(
                """
            Powered by [Moondream 2B](https://github.com/vikhyat/moondream).

            Upload a video and specify what to moderate. The app will process each frame and moderate any visual content that matches the prompt. For help, join the [Moondream Discord](https://discord.com/invite/tRUdpjDQfH).
            """
            )

            with gr.Row():
                with gr.Column():
                    # Input components
                    video_input = gr.Video(label="Upload Video")

                    detect_input = gr.Textbox(
                        label="What to Moderate",
                        placeholder="e.g. face, cigarette, gun, etc.",
                        value="face",
                        info="Moondream can moderate anything that you can describe in natural language",
                    )

                    process_btn = gr.Button("Process Video", variant="primary")

                    with gr.Accordion("Advanced Settings", open=False):
                        box_style_input = gr.Radio(
                            choices=[
                                "censor",
                                "bounding-box",
                                "hitmarker",
                                "sam",
                                "sam-fast",
                                "fuzzy-blur",
                                "pixelated-blur",
                                "intense-pixelated-blur",
                                "obfuscated-pixel",
                            ],
                            value="obfuscated-pixel",
                            label="Visualization Style",
                            info="Choose how to display moderations: censor (black boxes), bounding-box (red boxes with labels), hitmarker (COD-style markers), sam (precise segmentation), sam-fast (faster but less precise segmentation), fuzzy-blur (Gaussian blur), pixelated-blur (pixelated with blur), obfuscated-pixel (advanced pixelation with neighborhood averaging)",
                        )
                        preset_input = gr.Dropdown(
                            choices=[
                                "ultrafast",
                                "superfast",
                                "veryfast",
                                "faster",
                                "fast",
                                "medium",
                                "slow",
                                "slower",
                                "veryslow",
                            ],
                            value="medium",
                            label="Processing Speed (faster = lower quality)",
                        )
                        with gr.Row():
                            rows_input = gr.Slider(
                                minimum=1, maximum=4, value=1, step=1, label="Grid Rows"
                            )
                            cols_input = gr.Slider(
                                minimum=1,
                                maximum=4,
                                value=1,
                                step=1,
                                label="Grid Columns",
                            )

                        test_mode_input = gr.Checkbox(
                            label="Test Mode (Process first 3 seconds only)",
                            value=True,
                            info="Enable to quickly test settings on a short clip before processing the full video (recommended). If using the data visualizations, disable.",
                        )

                        test_duration_input = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=3,
                            step=1,
                            label="Test Mode Duration (seconds)",
                            info="Number of seconds to process in test mode",
                        )

                        gr.Markdown(
                            """
                        Note: Processing in test mode will only process the first 3 seconds of the video and is recommended for testing settings.
                        """
                        )

                        gr.Markdown(
                            """
                        We can get a rough estimate of how long the video will take to process by multiplying the videos framerate * seconds * the number of rows and columns and assuming 0.12 seconds processing time per detection.
                        For example, a 3 second video at 30fps with 2x2 grid, the estimated time is 3 * 30 * 2 * 2 * 0.12 = 43.2 seconds (tested on a 4090 GPU).
                        
                        Note: Using the SAM visualization style will increase processing time significantly as it performs additional segmentation for each detection. The sam-fast option uses a smaller model for faster processing at the cost of some accuracy.
                        """
                        )

                with gr.Column():
                    # Output components
                    video_output = gr.Video(label="Processed Video")
                    json_output = gr.Text(label="Detection Data Path", visible=False)

                    # About section under the video output
                    gr.Markdown(
                        """
                    ### Links:
                    - [GitHub Repository](https://github.com/vikhyat/moondream)
                    - [Hugging Face](https://huggingface.co/vikhyatk/moondream2)
                    - [Quick Start](https://docs.moondream.ai/quick-start)
                    - [Moondream Recipes](https://docs.moondream.ai/recipes)
                    """
                    )

        with gr.Tab("Analyze Results"):
            gr.Markdown("# Detection Analysis")
            gr.Markdown(
                """
            Analyze the detection results from processed videos. The analysis includes:
            - Basic detection statistics and patterns
            - Temporal and spatial distribution analysis
            - Size-based categorization
            - Screen region analysis
            - Detection density patterns
            """
            )

            with gr.Row():
                json_input = gr.File(
                    label="Upload Detection Data (JSON)",
                    file_types=[".json"],
                )
                analyze_btn = gr.Button("Analyze", variant="primary")

            with gr.Row():
                with gr.Column():
                    plot1 = gr.Image(
                        label="Detections Per Frame",
                    )
                    plot2 = gr.Image(
                        label="Detection Areas Distribution",
                    )
                    plot5 = gr.Image(
                        label="Detection Density Timeline",
                    )
                    plot6 = gr.Image(
                        label="Screen Region Analysis",
                    )

                with gr.Column():
                    plot3 = gr.Image(
                        label="Average Detection Area Over Time",
                    )
                    plot4 = gr.Image(
                        label="Detection Center Heatmap",
                    )
                    plot7 = gr.Image(
                        label="Detection Size Categories",
                    )
                    plot8 = gr.Image(
                        label="Temporal Pattern Analysis",
                    )

            stats_output = gr.Textbox(
                label="Statistics",
                info="Summary of key metrics and patterns found in the detection data.",
                lines=12,
                max_lines=15,
                interactive=False,
            )

        # with gr.Tab("Video Visualizations"):
        #     gr.Markdown("# Real-time Detection Visualization")
        #     gr.Markdown(
        #         """
        #     Watch the detection patterns unfold in real-time. Choose from:
        #     - Timeline: Shows number of detections over time
        #     - Gauge: Simple yes/no indicator for current frame detections
        #     """
        #     )

        #     with gr.Row():
        #         json_input_realtime = gr.File(
        #             label="Upload Detection Data (JSON)",
        #             file_types=[".json"],
        #         )
        #         viz_style = gr.Radio(
        #             choices=["timeline", "gauge"],
        #             value="timeline",
        #             label="Visualization Style",
        #             info="Choose between timeline view or simple gauge indicator"
        #         )
        #         visualize_btn = gr.Button("Visualize", variant="primary")

        #     with gr.Row():
        #         video_visualization = gr.Video(
        #             label="Detection Visualization",
        #             interactive=False
        #         )
        #         stats_realtime = gr.Textbox(
        #             label="Video Statistics",
        #             lines=6,
        #             max_lines=8,
        #             interactive=False
        #         )

    # Event handlers
    process_outputs = process_btn.click(
        fn=process_video_file,
        inputs=[
            video_input,
            detect_input,
            box_style_input,
            preset_input,
            rows_input,
            cols_input,
            test_mode_input,
            test_duration_input,
        ],
        outputs=[video_output, json_output],
    )

    # Auto-analyze after processing
    process_outputs.then(
        fn=create_visualization_plots,
        inputs=[json_output],
        outputs=[plot1, plot2, plot3, plot4, plot5, plot6, plot7, plot8, stats_output],
    )

    # Manual analysis button
    analyze_btn.click(
        fn=create_visualization_plots,
        inputs=[json_input],
        outputs=[plot1, plot2, plot3, plot4, plot5, plot6, plot7, plot8, stats_output],
    )

    # Video visualization button
    # visualize_btn.click(
    #     fn=lambda json_file, style: create_video_visualization(json_file.name if json_file else None, style),
    #     inputs=[json_input_realtime, viz_style],
    #     outputs=[video_visualization, stats_realtime],
    # )

if __name__ == "__main__":
    app.launch(share=True)
