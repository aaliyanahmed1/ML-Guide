# inference_rfdetr_nano.py
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from transformers import AutoImageProcessor, RfDetrForObjectDetection

# 1. Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Load RF-DETR Nano pretrained model
processor = AutoImageProcessor.from_pretrained("roboflow/rf-detr-large")
model = RfDetrForObjectDetection.from_pretrained("roboflow/rf-detr-large")
model.to(device)
model.eval()

# 3. Load input image
image_path = "test_image.jpg"  # replace with your image
image = Image.open(image_path).convert("RGB")

# 4. Preprocess image
inputs = processor(images=image, return_tensors="pt").to(device)

# 5. Run inference
with torch.no_grad():
    outputs = model(**inputs)
    
# 6. Post-process detections
results = processor.post_process_object_detection(
    outputs,
    threshold=0.5,  # confidence threshold
    target_sizes=[(image.height, image.width)]
)[0]

# 7. Visualize bounding boxes
plt.figure(figsize=(12, 8))
plt.imshow(image)

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [int(i) for i in box.tolist()]
    rect = patches.Rectangle(
        (box[0], box[1]),
        box[2] - box[0],
        box[3] - box[1],
        linewidth=2,
        edgecolor='r',
        facecolor='none'
    )
    plt.gca().add_patch(rect)
    plt.text(
        box[0], box[1] - 5,
        f"Class {label}: {score:.2f}",
        color='red', fontsize=10, backgroundcolor='white'
    )

plt.axis('off')
plt.show()
