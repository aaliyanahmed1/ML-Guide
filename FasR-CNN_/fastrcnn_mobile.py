import torch
import torchvision
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision.transforms as T

class FasterRCNN_Mobile:
    def __init__(self, num_classes=91, confidence_threshold=0.5):
        """
        Initialize Faster R-CNN with MobileNetV3 backbone
        Args:
            num_classes: Number of classes (91 for COCO)
            confidence_threshold: Detection confidence threshold
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.confidence_threshold = confidence_threshold
        
        # Load model with MobileNetV3 backbone
        self.model = fasterrcnn_mobilenet_v3_large_fpn(
            weights='DEFAULT',
            box_score_thresh=confidence_threshold,
            rpn_pre_nms_top_n_test=100,  # Reduce computations for mobile
            rpn_post_nms_top_n_test=100
        )
        self.model.to(self.device)
        self.model.eval()
        
        # Load COCO class names
        self.classes = self._load_coco_classes()
        
        # Image transforms
        self.transform = T.Compose([T.ToTensor()])
    
    def _load_coco_classes(self):
        """Load COCO class names"""
        from torchvision.datasets import CocoDetection
        return CocoDetection.classes
    
    def detect(self, image_path):
        """
        Detect objects in image
        Args:
            image_path: Path to input image
        Returns:
            boxes, labels, scores
        """
        # Load and transform image
        image = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            prediction = self.model(img_tensor)[0]
            
        # Get detections above threshold
        keep = prediction['scores'] > self.confidence_threshold
        boxes = prediction['boxes'][keep].cpu()
        labels = prediction['labels'][keep].cpu()
        scores = prediction['scores'][keep].cpu()
        
        return image, boxes, labels, scores
    
    def visualize(self, image, boxes, labels, scores, save_path=None):
        """
        Visualize detections on image
        Args:
            image: PIL Image
            boxes: Detected boxes
            labels: Class labels
            scores: Confidence scores
            save_path: Path to save visualization
        """
        plt.figure(figsize=(12, 8))
        plt.imshow(image)
        
        # Draw each detection
        for box, label, score in zip(boxes, labels, scores):
            # Create rectangle patch
            x1, y1, x2, y2 = box.numpy()
            rect = patches.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                linewidth=2, edgecolor='b', facecolor='none'
            )
            plt.gca().add_patch(rect)
            
            # Add label and score
            class_name = self.classes[label-1]  # COCO labels start at 1
            plt.text(
                x1, y1-5,
                f'{class_name}: {score:.2f}',
                color='blue',
                fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8)
            )
        
        plt.axis('off')
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()

def main():
    # Initialize detector
    detector = FasterRCNN_Mobile(confidence_threshold=0.5)
    
    # Detect objects
    image_path = "test_image.jpg"  # Replace with your image
    image, boxes, labels, scores = detector.detect(image_path)
    
    # Visualize results
    detector.visualize(image, boxes, labels, scores, "output_mobile.jpg")

if __name__ == "__main__":
    main()
