# Moondream Python Quickstart

Get started with Moondream's vision AI in Python in minutes.

## Setup

1. Choose your deployment:

   - **Local**: Install [Moondream Station](https://moondream.ai/station)
   - **Cloud**: Get an API key from [Moondream Cloud](https://moondream.ai/cloud)

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run Moondream Station locally or tweak the [main.py](quickstart/python/main.py) or run on the cloud using an [API key](https://moondream.ai/c/cloud/api-keys).

4. Run the example to see Moondream in action!

   ```bash
   python main.py
   ```

5. Try a full tutorial:

   Open the `detect_cars.ipynb` notebook in Jupyter and run all the cells to try caption, query, detect and point capabilities:

   ```bash
   jupyter notebook detect_cars.ipynb
   ```

## Note: 0.x vs 1.x client API

If you see `TypeError: ... got an unexpected keyword argument 'model'`,
you are running a 1.x `moondream` package with a pre-1.0 snippet.
Version 0.x loaded a local model file directly:

```python
# moondream 0.x only (no longer supported)
model = md.vl(model="<path-to-model-file>")
```

Since 1.0, `md.vl()` is endpoint-based — pass an API key and/or an
endpoint instead (see `main.py`):

```python
import moondream as md

# Local Station (see https://moondream.ai/station)
model = md.vl(endpoint="http://localhost:2020/v1")

# Cloud (see https://moondream.ai/c/cloud/api-keys)
# model = md.vl(api_key="<your-api-key>")
```

To run weights directly without Station, use the
[Hugging Face Transformers implementation](https://huggingface.co/vikhyatk/moondream2)
(`vikhyatk/moondream2`, see `sample.py`) or
[Moondream Server](https://moondream.ai/moondream-server).