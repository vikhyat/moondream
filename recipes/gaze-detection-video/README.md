# Gaze Detection Video Processor

> **⚠️ IMPORTANT:** This project currently uses Moondream 2B (2025-01-09 release) via the Hugging Face Transformers library. We will migrate to the official Moondream client
> libraries once they become available for this version.

## Table of Contents

- [Overview](#overview)
- [Sample Output](#sample-output)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
  - [Linux/macOS Installation](#linuxmacos-installation)
  - [Windows Installation](#windows-installation)
- [Usage](#usage)
- [Output](#output)
- [Troubleshooting](#troubleshooting)
- [Performance Notes](#performance-notes)
- [Dependencies](#dependencies)
- [Model Details](#model-details)
- [License](#license)

## Overview

This project uses the Moondream 2B model to detect faces and their gaze directions in videos. It processes videos frame by frame, visualizing face detections and gaze directions.

## Sample Output

|              Input Video              |              Processed Output               |
| :-----------------------------------: | :-----------------------------------------: |
| ![Input Video](https://github.com/parsakhaz/gaze-detection-video/blob/master/gif-input-sample.gif?raw=true) | ![Processed Output](https://github.com/parsakhaz/gaze-detection-video/blob/master/gif-output-sample.gif?raw=true) |

## Features

- Face detection in video frames
- Gaze direction tracking
- Real-time visualization with:
  - Colored bounding boxes for faces
  - Gradient lines showing gaze direction
  - Gaze target points
- Supports multiple faces per frame
- Processes all common video formats (.mp4, .avi, .mov, .mkv)
- Uses Moondream 2 (2025-01-09 release) via Hugging Face Transformers
  - Note: Will be migrated to official client libraries in future updates
  - No authentication required

## Prerequisites

1. Python 3.8 or later
2. CUDA-capable GPU recommended (but CPU mode works too)
3. FFmpeg installed on your system

## Installation

### Linux/macOS Installation

1. Install system dependencies:

   ```bash
   # Ubuntu/Debian
   sudo apt-get update && sudo apt-get install -y libvips42 libvips-dev ffmpeg

   # CentOS/RHEL
   sudo yum install vips vips-devel ffmpeg

   # macOS
   brew install vips ffmpeg
   ```

2. Clone and setup the project:
   ```bash
   git clone https://github.com/vikhyat/moondream.git
   cd moondream/recipes/gaze-detection-video
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

### Windows Installation

Windows setup requires a few additional steps for proper GPU support and libvips installation.

1. Clone the repository:

   ```bash
   git clone [repository-url]
   cd moondream/recipes/gaze-detection-video
   ```

2. Create and activate virtual environment:

   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. Install PyTorch with CUDA support:

   ```bash
   # For NVIDIA GPUs
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

4. Install libvips: Download the appropriate version based on your system architecture:

   | Architecture | VIPS Version to Download |
   | ------------ | ------------------------ |
   | 32-bit x86   | vips-dev-w32-all-8.16.0.zip |
   | 64-bit x64   | vips-dev-w64-all-8.16.0.zip |

   - Extract the ZIP file
   - Copy all DLL files from `vips-dev-8.16\bin` to either:
     - Your project's root directory (easier) OR
     - `C:\Windows\System32` (requires admin privileges)
   - Add to PATH:
     1. Open System Properties → Advanced → Environment Variables
     2. Under System Variables, find PATH
     3. Add the full path to the `vips-dev-8.16\bin` directory

5. Install FFmpeg:

   - Download from https://ffmpeg.org/download.html#build-windows
   - Extract and add the `bin` folder to your system PATH (similar to step 4) or to the project root directory

6. Install other dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Place your input videos in the `input` directory

   - Supported formats: .mp4, .avi, .mov, .mkv
   - The directory will be created automatically if it doesn't exist

2. Run the script:

   ```bash
   python gaze-detection-video.py
   ```

3. The script will:
   - Process all videos in the input directory
   - Show progress bars for each video
   - Save processed videos to the `output` directory with prefix 'processed\_'

## Output

- Processed videos are saved as `output/processed_[original_name].[ext]`
- Each frame in the output video shows:
  - Colored boxes around detected faces
  - Lines indicating gaze direction
  - Points showing where each person is looking

## Troubleshooting

1. CUDA/GPU Issues:

   - Ensure you have CUDA installed for GPU support
   - The script will automatically fall back to CPU if no GPU is available

2. Memory Issues:

   - If processing large videos, ensure you have enough RAM
   - Consider reducing video resolution if needed

3. libvips Errors:

   - Make sure libvips is properly installed for your OS
   - Check system PATH includes libvips

4. Video Format Issues:
   - Ensure FFmpeg is installed and in your system PATH
   - Try converting problematic videos to MP4 format

## Performance Notes

- GPU processing is significantly faster than CPU
- Processing time depends on:
  - Video resolution
  - Number of faces per frame
  - Frame rate
  - Available computing power

## Dependencies

- transformers (for Moondream 2 model access)
- torch
- opencv-python
- pillow
- matplotlib
- numpy
- tqdm
- pyvips
- accelerate
- einops

## Model Details

> **⚠️ IMPORTANT:** This project currently uses Moondream 2 (2025-01-09 release) via the Hugging Face Transformers library. We will migrate to the official Moondream client
> libraries once they become available for this version.

The model is loaded using:
