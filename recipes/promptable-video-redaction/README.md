# Promptable Video Redaction with Moondream

This tool uses Moondream 2B, a powerful yet lightweight vision-language model, to detect and redact objects from videos. Moondream can recognize a wide variety of objects, people,
text, and more with high accuracy while being much smaller than traditional models.

[Try it now.](https://huggingface.co/spaces/moondream/promptable-video-redaction)

## About Moondream

Moondream is a tiny yet powerful vision-language model that can analyze images and answer questions about them. It's designed to be lightweight and efficient while maintaining high
accuracy. Some key features:

- Only 2B parameters
- Fast inference with minimal resource requirements
- Supports CPU and GPU execution
- Open source and free to use
- Can detect almost anything you can describe in natural language

Links:

- [GitHub Repository](https://github.com/vikhyat/moondream)
- [Hugging Face](https://huggingface.co/vikhyatk/moondream2)
- [Build with Moondream](http://docs.moondream.ai/)

## Features

- Real-time object detection in videos using Moondream
- Multiple visualization styles:
  - Censor: Black boxes over detected objects
  - Bounding Box: Traditional bounding boxes with labels
  - Hitmarker: Call of Duty style crosshair markers
- Optional grid-based detection for improved accuracy
- Flexible object type detection using natural language
- Frame-by-frame processing with IoU-based merging
- Batch processing of multiple videos
- Web-compatible output format
- User-friendly web interface
- Command-line interface for automation

## Requirements

- Python 3.8+
- OpenCV (cv2)
- PyTorch
- Transformers
- Pillow (PIL)
- tqdm
- ffmpeg
- numpy
- gradio (for web interface)

## Installation

1. Clone this repository and create a new virtual environment

```bash
git clone https://github.com/vikhyat/moondream/blob/main/recipes/promptable-video-redaction
python -m venv .venv
source .venv/bin/activate
```

2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Install ffmpeg:
   - On Ubuntu/Debian: `sudo apt-get install ffmpeg libvips`
   - On macOS: `brew install ffmpeg`
   - On Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html)
     > Downloading libvips for Windows requires some additional steps, see [here](https://docs.moondream.ai/quick-start)

## Usage

### Web Interface

1. Start the web interface:

```bash
python app.py
```

2. Open the provided URL in your browser

3. Use the interface to:
   - Upload your video
   - Specify what to censor (e.g., face, logo, text)
   - Adjust processing speed and quality
   - Configure grid size for detection
   - Process and download the censored video

### Command Line Interface

1. Create an `inputs` directory in the same folder as the script:

```bash
mkdir inputs
```

2. Place your video files in the `inputs` directory. Supported formats:

   - .mp4
   - .avi
   - .mov
   - .mkv
   - .webm

3. Run the script:

```bash
python main.py
```

### Optional Arguments:

- `--test`: Process only first 3 seconds of each video (useful for testing detection settings)

```bash
python main.py --test
```

- `--preset`: Choose FFmpeg encoding preset (affects output quality vs. speed)

```bash
python main.py --preset ultrafast  # Fastest, lower quality
python main.py --preset veryslow   # Slowest, highest quality
```

- `--detect`: Specify what object type to detect (using natural language)

```bash
python main.py --detect person     # Detect people
python main.py --detect "red car"  # Detect red cars
python main.py --detect "person wearing a hat"  # Detect people with hats
```

- `--box-style`: Choose visualization style

```bash
python main.py --box-style censor     # Black boxes (default)
python main.py --box-style bounding-box       # Bounding box-style boxes with labels
python main.py --box-style hitmarker  # COD-style hitmarkers
```

- `--rows` and `--cols`: Enable grid-based detection by splitting frames

```bash
python main.py --rows 2 --cols 2   # Split each frame into 2x2 grid
python main.py --rows 3 --cols 3   # Split each frame into 3x3 grid
```

You can combine arguments:

```bash
python main.py --detect "person wearing sunglasses" --box-style bounding-box --test --preset "fast" --rows 2 --cols 2
```

### Visualization Styles

The tool supports three different visualization styles for detected objects:

1. **Censor** (default)

   - Places solid black rectangles over detected objects
   - Best for privacy and content moderation
   - Completely obscures the detected region

2. **Bounding Box**

   - Traditional object detection style
   - Red bounding box around detected objects
   - Label showing object type above the box
   - Good for analysis and debugging

3. **Hitmarker**
   - Call of Duty inspired visualization
   - White crosshair marker at center of detected objects
   - Small label above the marker
   - Stylistic choice for gaming-inspired visualization

Choose the style that best fits your use case using the `--box-style` argument.

## Output

Processed videos will be saved in the `outputs` directory with the format: `[style]_[object_type]_[original_filename].mp4`

For example:

- `censor_face_video.mp4`
- `bounding-box_person_video.mp4`
- `hitmarker_car_video.mp4`

The output videos will include:

- Original video content
- Selected visualization style for detected objects
- Web-compatible H.264 encoding

## Notes

- Processing time depends on video length, grid size, and GPU availability
- GPU is strongly recommended for faster processing
- Requires sufficient disk space for temporary files
- Detection quality varies based on video quality and Moondream's ability to recognize the specified object
- Grid-based detection impacts performance significantly - use only when needed
- Web interface shows progress updates and errors
- Choose visualization style based on your use case
- Moondream can detect almost anything you can describe in natural language
