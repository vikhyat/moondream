# Promptable Content Moderation with Moondream

Welcome to the future of content moderation. This tool uses Moondream 2B, a powerful yet lightweight vision-language model, to moderate content in videos. Moondream allows you to detect and moderate content using natural language prompts.

[Try it now.](https://huggingface.co/spaces/moondream/content-moderation)

## Features

- Content moderation through natural language prompts
- Multiple visualization styles
- Intelligent scene detection and tracking:
  - DeepSORT tracking with scene-aware reset
  - Persistent moderation across frames
  - Smart tracker reset at scene boundaries
- Optional grid-based detection for improved accuracy on complex scenes
- Frame-by-frame processing with IoU-based merging
- Web-compatible output format
- Test mode (process only first X seconds)
- Advanced moderation analysis with multiple visualization plots

## Examples

| Example Outputs |
|------|
| ![Demo](./examples/clip-cig.gif)     |
| ![Demo](./examples/clip-gu.gif)      |
| ![Demo](./examples/clip-conflag.gif) |

## Requirements

### Python Dependencies

For Windows users, before installing other requirements, first install PyTorch with CUDA support:

```bash
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

Then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

### System Requirements

- FFmpeg (required for video processing)
- libvips (required for image processing)

Installation by platform:

- Ubuntu/Debian: `sudo apt-get install ffmpeg libvips`
- macOS: `brew install ffmpeg libvips`
- Windows:
  - Download FFmpeg from [ffmpeg.org](https://ffmpeg.org/download.html)
  - Follow [libvips Windows installation guide](https://docs.moondream.ai/quick-start)

### Hardware Requirements

- GPU recommended for faster processing (CUDA compatible)
- Minimum 8GB RAM
- Storage space for temporary files and output videos

## Installation

1. Clone this repository and create a new virtual environment:

```bash
git clone https://github.com/vikhyat/moondream/blob/main/recipes/promptable-video-redaction
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

3. Install ffmpeg and libvips:
   - On Ubuntu/Debian: `sudo apt-get install ffmpeg libvips`
   - On macOS: `brew install ffmpeg`
   - On Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html)

> Downloading libvips for Windows requires some additional steps, see [here](https://docs.moondream.ai/quick-start)

## Usage

The easiest way to use this tool is through its web interface, which provides a user-friendly experience for video content moderation.

### Web Interface

1. Start the web interface:

```bash
python app.py
```

2. Open the provided URL in your browser (typically <http://localhost:7860>)

3. Use the interface to:
   - Upload your video file
   - Specify content to moderate (e.g., "face", "cigarette", "gun")
   - Choose redaction style (default: obfuscated-pixel)
   - OPTIONAL: Configure advanced settings
     - Processing speed/quality
     - Grid size for detection
     - Test mode for quick validation (default: on, 3 seconds)
   - Process the video and download results
   - Analyze detection patterns with visualization tools

## Output Files

The tool generates two types of output files in the `outputs` directory:

1. Processed Videos:
   - Format: `[style]_[content_type]_[original_filename].mp4`
   - Example: `censor_inappropriate_video.mp4`

2. Detection Data:
   - Format: `[style]_[content_type]_[original_filename]_detections.json`
   - Contains frame-by-frame detection information
   - Used for visualization and analysis

## Technical Details

### Scene Detection and Tracking

The tool uses advanced scene detection and object tracking:

1. Scene Detection:
   - Powered by PySceneDetect's ContentDetector
   - Automatically identifies scene changes in videos
   - Configurable detection threshold (default: 30.0)
   - Helps maintain tracking accuracy across scene boundaries

2. Object Tracking:
   - DeepSORT tracking for consistent object identification
   - Automatic tracker reset at scene changes
   - Maintains object identity within scenes
   - Prevents tracking errors across scene boundaries

3. Integration Benefits:
   - More accurate object tracking
   - Better handling of scene transitions
   - Reduced false positives in tracking
   - Improved tracking consistency

## Best Practices

- Use test mode for initial configuration
- Enable grid-based detection for complex scenes
- Choose appropriate redaction style based on content type:
  - Censor: Complete content blocking
  - Blur styles: Less intrusive moderation
  - Bounding Box: Content review and analysis
- Monitor system resources during processing
- Use appropriate processing quality settings based on your needs

## Notes

- Processing time depends on video length, resolution, GPU availability, and chosen settings
- GPU is strongly recommended for faster processing
- Grid-based detection increases accuracy but requires more processing time (each grid cell is processed independently)
- Test mode processes only first X seconds (default: 3 seconds) for quick validation
