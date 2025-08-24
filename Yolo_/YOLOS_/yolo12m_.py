# minimal_yolo_inference.py

from ultralytics import YOLO
import cv2

# 1. Load a pretrained YOLO model (nano/small/medium/large)
model = YOLO("yolo12m.pt")  #
# 2. Load an image
image_path = "image.jpg"
image = cv2.imread(image_path)

# 3. Perform inference
results = model(image)

# 4. Print predictions (class, confidence, bbox)
for result in results:
    boxes = result.boxes
    for box in boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].tolist()
        print(f"Class: {cls}, Confidence: {conf:.2f}, BBox: {xyxy}")

# 5. Show/save annotated image
annotated_image = results[0].plot()  # draws boxes and labels
cv2.imwrite("output.jpg", annotated_image)
cv2.imshow("YOLO Detection", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
