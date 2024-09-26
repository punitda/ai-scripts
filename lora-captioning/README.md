# Lora Captioning

Simple script to caption images for training a LoRA model. It uses [Florence-2-large](https://huggingface.co/microsoft/Florence-2-large) to caption images in folder and then saves the captions to a file.

Idea picked up from [ai-toolkit](https://github.com/ostris/ai-toolkit) repo

## Usage

1. Activate virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies

```bash
pip install -r lora-captioning/requirements.txt
```

3. Run the script

```bash
python lora-captioning/captioning.py
# Enter the path to the folder with images when prompted
```
