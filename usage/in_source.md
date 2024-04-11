**Using from source**

Clone this repository
```bash
git clone https://github.com/vikhyat/moondream.git
```
```bash
cd moondream
```

Install dependencies.
```bash
pip install -r requirements.txt
```

Use `gradio_demo.py` script to start a Gradio interface for the model.

```bash
python gradio_demo.py
```

Use `--model` flag for models on your local machine.

```bash
python gradio_demo.py --model MODEL_PATH
```

`sample.py` provides a CLI interface for running the model. When the `--prompt` argument is not provided, the script will allow you to ask questions interactively.

```bash
python sample.py --image [IMAGE_PATH] --prompt [PROMPT]
```

`webcam_gradio_demo.py` provides a Gradio interface for the model that uses your webcam as input and performs inference in real-time.

```bash
python webcam_gradio_demo.py
```
