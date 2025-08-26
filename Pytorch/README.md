# PyTorch 

This folder contains well‑commented scripts for object detection and
quick intros to core PyTorch libraries.

- `_inference.py`
  - object detection inference using a hardcoded torchvision model
    (Faster R‑CNN ResNet50 FPN). Takes an image path and writes an annotated
    image. Keep it simple: no CLI beyond `--image` and optional `--output`.
- `_training.py`
  - object detection training on a COCO‑format dataset using
    Faster R‑CNN. Constants at the top define paths and hyperparameters.
- `torch_.py`
  - Basic PyTorch: tensors + a tiny classifier training example.
- `torchvision_.py`
  - TorchVision intro: load a pretrained ResNet, preprocess an image, tiny
    one‑epoch CIFAR10 training demo.
- `torchaudio_.py`
  - TorchAudio intro: synthesize audio, extract Mel spectrogram, tiny
    classifier.
- `requirements.txt`
  - Required packages for the examples in this folder.

## Clone & Run

```bash
# 1) Clone the repo
git clone https://github.com/aaliyanahmed1/ML-Guide

# 2) Enter this folder
cd ML-Guide/Pytorch

# 3) (Optional) Create venv, then install deps
# python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 4) Run minimal inference (edit the image path in the command)
python _inference.py --image path/to/image.jpg --output result.jpg

# 5) Run minimal training (edit constants at top of the file, then)
python _training.py

# Library intros (optional)
python torch_.py
python torchvision_.py
python torchaudio_.py
```

## Notes

- `_inference.py` uses a COCO‑pretrained Faster R‑CNN for quick results.
- `_training.py` expects COCO JSON annotations and image directories; set
  `IMAGES_DIR`, `ANNOTATIONS_JSON`, etc. at the top of the script.
- All scripts are intentionally minimal and documented for clarity.
