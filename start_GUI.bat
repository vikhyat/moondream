@echo off

set HF_HOME=huggingface

call venv\Scripts\activate
python gradio_demo.py

pause
