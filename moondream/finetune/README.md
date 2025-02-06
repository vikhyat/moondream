# Finetuning Moondream 2B

This readme will walk you through the process of finetuning the text and region encoders of the Moondream 2B model. 

> Make sure to run all commands from the root directory of the project.

## Initial Setup

### Clone and Setup Environment
```bash
git clone https://github.com/vikhyat/moondream
cd moondream
python -m venv .venv
source .venv/bin/activate
```

### Install Dependencies
```bash
# Install base requirements
pip install -r requirements.txt

# Install finetuning specific dependencies
pip install safetensors datasets bitsandbytes tqdm wandb einops
```

## Downloading the Base Model

Download `model.safetensors` from the [Hugging Face repository](https://huggingface.co/vikhyatk/moondream2/tree/main) and place it in the `models` directory as `moondream_base.safetensors`.

```bash
# Create models directory
mkdir -p models

# Download it using curl (run from root moondream directory)
wget https://huggingface.co/vikhyatk/moondream2/resolve/main/model.safetensors
```

## Weights & Biases

We use Weights & Biases (wandb) to track finetuning progress.

To set it up to track your runs, use `wandb login`

This will take you through creating an account if you don't have one setup already. Enter your API key and you're ready to go.

## Finetuning the Text Encoder 

For this example, we will be teaching Moondream to describe images. 

Given the prompt: 
`\n\nQuestion: Describe this image.\n\nAnswer:`

We return a more detailed caption of the image then you would get from the base model.

1. Double check that you've updated MODEL_PATH to point to the base moondream model in `moondream/finetune/finetune_text.py`
2. Double check that the save path ends in `.safetensors`, otherwise the run will fail.
> Navigate to line 150 in `moondream/finetune/finetune_text.py`,
``` # Add save path
    save_file(
        model.state_dict(),
        "", // update this line ex: "models/moondream_text_finetuned.safetensors"
    )
```

### Start Text Finetuning
```bash
python -m moondream.finetune.finetune_text
```

The process will output a finetuned version of Moondream into your save path. Example output: `models/moondream_text_finetuned.safetensors`

### Test the Finetuned Text Encoder

You can test the finetuned models performance with the following command (run from root moondream directory).

This will return the caption of the image.

```bash
# Remember to update the paths
python -m moondream.torch.sample --model [FINETUNED_MODEL_PATH] --image "[DATASET_DIRECTORY]/test/[IMAGE_NAME]" --prompt "\n\nQuestion: Describe this image.\n\nAnswer:" --endpoint query
```

## Finetuning the Region Encoder

For this example, we will be teaching Moondream to detect railroad cracks in images of a railway. 

Our dataset trains our model such that,

Given the prompt: 
`\n\nDetect: crack\n\n`

We are returned the coordinates of a detected crack in the following format:
```{'objects': [{'x_min': [X_MIN], 'y_min': [Y_MIN], 'x_max': [X_MAX], 'y_max': [Y_MAX]}]}```

### Setup Dataset Dependencies

1. Visit https://universe.roboflow.com/research-zwl99/railwayvision
2. Download dataset in COCO JSON format into relevant directory (ex: `datasets`)
3. Update path to `annotation_file` (line 169) & `img_dir` (line 170) in `finetune_region.py` to point at the dataset 
- `annotation_file` should point to `<dataset_directory>/train/_annotations.coco.json`
- `img_dir` should point to `<dataset_directory>/train/`
4. Double check that you've updated MODEL_PATH to point to the base moondream model in `moondream/finetune/finetune_region.py`
5. Double check that the save path ends in `.safetensors`, otherwise the run will fail.
> Navigate to line 262 in `moondream/finetune/finetune_region.py`
``` # Add save path
    save_file(
        model.state_dict(),
        "", // update this line ex: "models/moondream_region_finetuned.safetensors"
    )
```

### Start Region Finetuning
```bash
python -m moondream.finetune.finetune_region
```

The process will output a finetuned version of Moondream into your save path. Example output: `models/moondream_region_finetuned.safetensors`