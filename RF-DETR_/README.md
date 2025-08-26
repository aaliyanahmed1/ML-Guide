# RF-DETR (Roboflow Inference) — Object Detection

Minimal RF-DETR inference examples using the Roboflow Inference SDK. Each script
loads a specific RF-DETR variant, runs inference on a local image, prints
results, and saves a simple visualization.

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

`requirements.txt` includes:
- torch, torchvision (common CV deps)
- inference (Roboflow Inference SDK)
- pillow, matplotlib, numpy, pycocotools, tqdm

## Clone & Run

```bash
# 1) Clone your repository
git clone <YOUR_REPO_URL>

# 2) Enter the RF-DETR folder
cd ML-Guide/RF-DETR_

# 3) (Recommended) Create venv, then install requirements
# python -m venv .venv && source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 4) Edit the script to set the IMAGE_PATH (and MODEL_ID if desired)
#    e.g., open inference_rfdetr.py and update IMAGE_PATH at the top

# 5) Run an example
python inference_rfdetr.py

# Or run a specific model variant demo (edit image_path inside the script first)
python rfdetr-nano.py
python rfdetr-small.py
python rfdetrbase.py
python rfdetrlarge.py
```

## Files in this folder

- `inference_rfdetr.py`
  - End-to-end, timed inference using Roboflow SDK: loads model via
    `get_model("rfdetr-base/1")` by default, reads a local image, prints timings
    and detections, and saves an annotated image.
  - Inputs: update `IMAGE_PATH`, `MODEL_ID`, `CONFIDENCE_THRESHOLD`, `OUTPUT_IMAGE` at the top.
  - Outputs: console logs (timing + detections) and an annotated image.

- `rfdetr-nano.py`
  - Minimal example loading `rfdetr-nano/1` via Roboflow SDK and visualizing
    boxes with class ids + confidences.
  - Input: edit `image_path` in the script.
  - Output: matplotlib window with drawn boxes.

- `rfdetr-small.py`
  - Minimal example loading `rfdetr-small/1`.
  - Input/Output same as above.

- `rfdetrbase.py`
  - Minimal example loading `rfdetr-base/1`.
  - Input/Output same as above.

- `rfdetrlarge.py`
  - Minimal example loading `rfdetr-large/1`.
  - Input/Output same as above.

- `train_rfdetr.py`
  - (Placeholder/minimal) training entry; customize for your workflow if needed.

## Input / Output format (scripts)

- Input: a local image path (e.g. `./test_image.jpg`)
- Output (printed):
  - number of detections
  - one line per detection: `(x1, y1, x2, y2) label_id=<int> score=<float>`
- Output (visual):
  - annotated image saved (for `inference_rfdetr.py`) or shown via matplotlib

## Why is torch imported in scripts?

- We use `torch` primarily to report device availability (CPU/GPU) and to
  handle tensors after converting predictions to a consistent format. The
  Roboflow Inference SDK itself runs fine without pushing tensors to GPU in
  these minimal examples.

## Quick Start (alternative)

1) Edit the image path and model id in the script you want to run
   (e.g., `inference_rfdetr.py`).

2) Run inference:

```bash
python inference_rfdetr.py
```

You should see console timings and an annotated image saved to disk.