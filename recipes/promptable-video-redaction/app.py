#!/usr/bin/env python3
import gradio as gr
import os
from main import load_moondream, process_video
import shutil
import torch

# Get absolute path to workspace root
WORKSPACE_ROOT = os.path.dirname(os.path.abspath(__file__))

# Check CUDA availability
print(f"Is CUDA available: {torch.cuda.is_available()}")
# We want to get True
print(f"CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
# GPU Name

# Initialize model globally for reuse
print("Loading Moondream model...")
model, tokenizer = load_moondream()


def process_video_file(
    video_file, detect_keyword, box_style, ffmpeg_preset, rows, cols, test_mode
):
    """Process a video file through the Gradio interface."""
    try:
        if not video_file:
            raise gr.Error("Please upload a video file")

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
                detect_keyword,
                test_mode=test_mode,
                ffmpeg_preset=ffmpeg_preset,
                rows=rows,
                cols=cols,
                box_style=box_style,
            )

            # Verify output exists and is readable
            if not output_path or not os.path.exists(output_path):
                print(f"Warning: Output path {output_path} does not exist")
                # Try to find the output based on expected naming convention
                expected_output = os.path.join(
                    outputs_dir, f"{box_style}_{detect_keyword}_{video_filename}"
                )
                if os.path.exists(expected_output):
                    output_path = expected_output
                else:
                    # Try searching in outputs directory for any matching file
                    matching_files = [
                        f
                        for f in os.listdir(outputs_dir)
                        if f.startswith(f"{box_style}_{detect_keyword}_")
                    ]
                    if matching_files:
                        output_path = os.path.join(outputs_dir, matching_files[0])
                    else:
                        raise gr.Error("Failed to locate output video")

            # Convert output path to absolute path if it isn't already
            if not os.path.isabs(output_path):
                output_path = os.path.join(WORKSPACE_ROOT, output_path)

            print(f"Returning output path: {output_path}")
            return output_path

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


# Create the Gradio interface
with gr.Blocks(title="Promptable Video Redaction") as app:
    gr.Markdown("# Promptable Video Redaction with Moondream")
    gr.Markdown(
        """
    [Moondream 2B](https://github.com/vikhyat/moondream) is a lightweight vision model that detects and visualizes objects in videos. It can identify objects, people, text and more.

    Upload a video and specify what to detect. The app will process each frame and apply your chosen visualization style. For help, join the [Moondream Discord](https://discord.com/invite/tRUdpjDQfH).
    """
    )

    with gr.Row():
        with gr.Column():
            # Input components
            video_input = gr.Video(label="Upload Video")
            detect_input = gr.Textbox(
                label="What to Detect",
                placeholder="e.g. face, logo, text, person, car, dog, etc.",
                value="face",
                info="Moondream can detect anything that you can describe in natural language",
            )
            process_btn = gr.Button("Process Video", variant="primary")

            with gr.Accordion("Advanced Settings", open=False):
                box_style_input = gr.Radio(
                    choices=["censor", "bounding-box", "hitmarker"],
                    value="censor",
                    label="Visualization Style",
                    info="Choose how to display detections",
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
                        minimum=1, maximum=4, value=1, step=1, label="Grid Columns"
                    )

                test_mode_input = gr.Checkbox(
                    label="Test Mode (Process first 3 seconds only)",
                    value=True,
                    info="Enable to quickly test settings on a short clip before processing the full video (recommended)",
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
                """
                )

        with gr.Column():
            # Output components
            video_output = gr.Video(label="Processed Video")

            # About section under the video output
            gr.Markdown(
                """
            ### Links:
            - [GitHub Repository](https://github.com/vikhyat/moondream)
            - [Hugging Face](https://huggingface.co/vikhyatk/moondream2)
            - [Python Package](https://pypi.org/project/moondream/)
            - [Moondream Recipes](https://docs.moondream.ai/recipes)
            """
            )

    # Event handlers
    process_btn.click(
        fn=process_video_file,
        inputs=[
            video_input,
            detect_input,
            box_style_input,
            preset_input,
            rows_input,
            cols_input,
            test_mode_input,
        ],
        outputs=video_output,
    )

if __name__ == "__main__":
    app.launch(share=True)
