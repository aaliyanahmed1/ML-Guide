# Machine Learning hands on Guide

Hi , in this guide we are going to deeply learn about machine learning’s fundamental concepts to core advance frameworks, model selection, benchmarking, fine-tuning and custom training and so on. From prototyping to production level real-time. deployments, from computer systems /servers to edge devices cross platform deployments. This guide will cover all the topics that are essentially needed to become zero-to-hero in Machine learning. In this documentation we will be focusing mainly on the object detection branch hat belongs to Deep learning . 
 
AI is simply ability to think and act like humans.  There are its branches  
Machine Leaning > Deep Learning > Computer Vision. 
 

**Machine Leaning**: A branch of AI in which machines learn from labelled data patterns instead of fixed rules-based systems . 
Three main types of Machine Learning: 

**1: Supervised Learning 2: Unsupervised Learning 3: Reinforcement Leaning.**

Deep Learning: Its branch of machine learning that is  based on the special type of neural networks that learns complex patterns from data.it mainly falls under supervised learning where models are trained on labeled datasets to learn hte mapping between inputs and outputs.
one of the special branch of deep learing is Computer vision that uses Convolutional Neural Networks to learn complex patterns from data and perform predictions in different enviroment efficiently in real time deployments as well .in computer vision, images are processed to extract meaningful features, which can then be used for various tasks such as classification, segmentation, and one of the most important applications—object detection, where models not only recognize objects but also locate them within the image.in this guide we will deeply learn about object detection.
 


## Object detection
 .It is a specific branch of computer vision that reads the images and detect the desired objects inside of the image along with their areas ( x,y coordinates). Like if you input an image to the object detection model [YOLO, RF-DTR etc(we will discuss next) ].  it won't only tell that there is object like cat\dog in the image. but also tells where they are by drawing bounding boxes on the areas and mentioning the class (name of object). 

image input > model> inference > detection > output 

![OBject Detection](/images/onnxpy.png)   

Like on this image we can see bounding boxes and the names of the objects like person and vehicle(bus) etc.. Thats the whole things. that is called object detection. It finds object and labels objects within an image. 

This was the verbal introduction if the object detection . 
Now move towards all the technical details and requirements needs to develop, select , fine-tune and train the object detection model. From dataset collection to preprocessing. 

**Object detection model selection**

hen selecting the model for the object detection tasks first, we must thoroughly do research according to requirements and use case like where we want to deploy it and what we want to detect from it so this is the must check before moving forward towards data-preprocessing and fine-tuning.  For example. you are developing a real-time security surveillance application that needs to detect events from the images under 50<ms . Then models speed shouldn't be compromised at any cost.   

On the other hand if you are developing medical imaging system to analyze medical reports like X-rays, MRIs and CT-scans  etc.  to detect deceases then accuracy must be the top check . No any minor compromise on the accuracy no matter how long model takes to detect because here we the use case isnt real-time so we can leverage the speed but not accuracy. But balance must be there in all cases. 

***Model performance Evaluation***
Now let see what are the evaluation steps to compare the speeds and accuracies of different models and select the best one according to the requirements. 

Speed metrices. 

Latency: how long the model takes to process one image\frame (milliseconds). lower latency mean faster processing. 

FPS: Frames per second how many frames model can process in 1 second. Valid for the real time applications. 

Model size: models' variants like nano, medium and large and the difference in speed and accuracy the tradeoffs among them and what fits in the required use case. 

Accuracy metrices : 

1: **Mean Average precision (mAP)**.it evaluates how precisely model detects objects among crowded frames. it measures the acuracy of the model in identifying and localizing objects withan an image. it combines precision( the proportion of correctly identified objects among all predicted objects) and recall( the proportion of correctly identified objects among all the actual objects).it gives a single score that ashows how well model finds objects and how well it avoids false positives. a higher mAP measn the model is more reliable and consistent.it helps to compare different models amd select according to the use case and requirment.

2: **Recall** The ratio of correctly predicted positive detections to all actual objects present.out of all real objects present, ow manydid the model successfully  detect high recall = fewer false negatives . it evaluates the models ability to capture every possible object without missing them.Recall is calculated by dividing true positives by the sum of true positives and false negatives . it ensures that even subtle and partially visible objects are not overlooked.higher recall make the model reliable for the scenerios where missing object is critical like medical imaging and security survellinece deployments.

3 :**Precision** t measures the accuracy of a model’s positive predictions, indicating the proportion of items predicted as positive that were actually correct. It is calculated as True Positives / (True Positives + False Positives). High precision means the model has few false positives. This metric helps you understand how trustworthy your model is when it predicts a specific outcome.

4:**IoU** it measures how much the predicted object bounding box overlaps with the real( ground truths) box.its the ratio between
overlap area/total combined area.Higher IOU = better prediction accuracy by the model.it compares the difference between  ground truths and predictions.
making visible the accuracy of the model. as visible in the image below green box is ground truth actual object area and the red box is predicted area by model so we can see slightly difference in overlapping of the boxes this visibly shows the accuracy of the model.

![IOU ](/images/IOU__.png)

certain threshold is set for predicting the class with accuracy.

![IOU_threshold](/images/IOU_THR.png)


These are the steps needs to consider and critically evaluate before selecting finalizing the model for the system. 

**Data-Preprocessing**
this is the most important and critical part of whole Machine Learning system. Whole performance of the model depends upon the dataset on which it's been trained. There is famous saying (garbage in garbage out). dataset must be cleaned well balanced and must cover all the features that are required, and systems needs to detect.

### Steps for Data-Preprocessing

Dataset is split into 2 folders: Train\Test. 
 
Train: this folder contains the larger amount of the dataset and models learns patterns and features from the images and annotations. 
 
Test: this contains the final set of images that model hasn't seen before. It used after training to check how accurate the model has performed in Realtime scenarios,simply testing the model on new images that were not incluided in the train dataset  on whioch labels were drawn.

#### Data-Cleaning
Removing duplicates to remove the wrong information, incorrect annotations and irrelevant images that are not required because model training requires resources like GPU and memory for data-saving so be sure to manage all the resources efficiently. 

#### Image resizing
Resizing all the images to a uniform size that is required by the selected model for the input. 

#### Data-Augmentation
This step of Data-preprocessing plays vital role in model performance and generalization.  this step includes the rotation, flipping(horizontal\vertical) to increase the diversity. Scaling: zooming in\out to stimulate at different distances. Brightnessadjustment:  this step changs the brightness levels of the images to stimulate the different conditions like sunlight, cloudy weather o any lighting changes that causes the colors (pixels) difference. By training model on the images varying brightness, it learns to recognize objects accurately regardless of environmental changes. Thís helps reduce the false positive and improves the model's performance for every condition.

#### Formats checking/conversion
Every model has its own specific annotation format for reading labels.so the annotations must be in it like hee are some examples . 

YOLO models use TXT file as label file containing the (class+bounding boxes x ,y coordinates ) for each image file and names of  both files image and label file should be exact same at all  
like eg : “ Image-1.jpg = image-1.txt” 

![Yolo format lables file](/images/yololabel.png)


****Detectron2, Faster R-CNN (PyTorch), Mask R-CNN (PyTorch), RF-DETR.****  

These models take JSON files as annotations .this file contains the metadata of the dataset including the information of each image (filename, size), the objects in each image(bounding boxes, categories of the object like class) and also the list of all objects\classes. Annotations are linked to the corresponding images using ids. 

![json file struture](/images/examplee.png)


#### Data preprocessing Platform. 
[Roboflow](https://roboflow.com/)

It's a web-based tool that has functionality to organize the data, preprocess including augmentation and format conversion among all the different models.  it allows users to upload their dataset and annotate it, augment it. select the model and it will generate well balanced dataset including the Train &Test.It allows you to train model in it as well on some free credits . and then you can choose the" paid version".   

![](/images/roboflow.png)
![](/images/code_snipper.png)

It guides you through all the necessary steps you need to complete and then generates an API key for direct dataset integration via API or as a downloadable .zip file. 

Here you can select the  model format . 
![](/images/format.png)


### Dataset collection

Now comes the main part that where to find and get the dataset to train model , so the universal platform where multiple datasets are available is 
[Kaggkle](https://www.kaggle.com/). its widely used and most of teh general datasets are avaiable on it for free.( check for the license of usage for each).

****Video-Frames****
In case if dataset isn't avaiable on this platform then we have second option.
fetching frames from videos and real-time recordings and then defining the obejects names. lets assume you are coolecting dataset for any company that has some products and they want to count allof them on the last stage of converyer belt to count the production of units  .so object can be anything special and not available, so then it comes that way of collecting dataset from the video recordings . just taking the video of those products where they are eg on the conveyer belt and then extracting framesfrom it using program as well (PYTHON).

This is the simple Python code snippet that can be used to extract frames from the videos to collect the dataset.

```python
import cv2
import os

# Set your variables here:
video_path = "input_video.mp4"
num_frames_to_extract = 5
output_folder = "frames"

os.makedirs(output_folder, exist_ok=True)
cap = cv2.VideoCapture(video_path)
count = 0

while count < num_frames_to_extract:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imwrite(f"{output_folder}/frame_{count}.jpg", frame)
    print(f"Saved frame {count}")
    count += 1

cap.release()
print("Done extracting frames.")
```


After you have collected your dataset images, you can use Roboflow to annotate them and convert the dataset into the format required by your model.

[Tutorial](https://youtu.be/Dk-6MCQ9j-c?si=dIzQyNsWWxoysQLV)
 complete guide how to prepare the dataset and getting it ready for the training.


### Training
This steps is similar to training any human baing for doing a specific task. so like that AI models are also trained  on the specific dataset to perform the  predictions besed on the patterns learn from the dataset .to start the training firstly we prepare dataset(discussed earlier). then selecting an specific architeure of the model.we can do this by fine-tuning an existing pre-trained model (YOLO,RT-DTER etc) or by  defining a custom architeucture of the model using frameworks like Pytorch, Tensorflow ,Keras etc .these framewords provide libraries and packages to defien ,train,test and evaluate models.
when setting up for the training, hyperparameters are configured to control and manage the learning process .they influnce how  the training progesses and they directly impacts the model's perforamnce and efficiency. like how, hyperparameters determines how quickly andeffectively model learns from patterns of the data.hyperparameters are epcohs,learning rate,batchsize,momentum and weight decay etc. lets have a simple overview of each and see what is role of each hyperparameter for effecting the training process at all.

**Learning Rate**  very cruicial unit for the training process like it covers the 70% of the model efficieny during training process becasue it defines the steps models needs to take to learn patterns from the data.it controls how much model's weights are updated during training .too high can unstable the model training and too low can slowdown the training .

**batch Size** Number of random samples from the dataset are  processed before model updates it weights .

***Epochs***  Number of times entire training datasets is passed through the model . more  epochs let the model learn better . more epochs more itreration over the dataset and more detail learning.

**Momentum** used in optimizers to accelarate  in consisten directions improving convergence speed.

**Weight-Decay** A regularization term added to prevent overfitting by penalizing large weights.


### Pytorch
Its an open-source framework that provides tools  to build ,train, and fine-tune models(neural networks).it provides each steps we need to make a working model .it has vast netowrk of libraries.
torch for core operations and inference applciations .
torchvision for computer vision image processing work .it has preloaded datasets that we can just import from library and the pretrained models that can be loaded to finetune them and also it provides the structure for defining custom models for differennt tasks . 
torchaudio for audio processing and torchtext for natural language processing tasks.

Minimal implementation of the pytorch code for fine-tuning(training) object detection model on custom dataset.

[Training](https://github.com/aaliyanahmed1/ML-Guide/blob/main/Pytorch/_training.py)

and this one is for inference 

[Inference](https://github.com/aaliyanahmed1/ML-Guide/blob/main/Pytorch/_inference.py)

**code implementation example**
[torch](https://github.com/aaliyanahmed1/ML-Guide/blob/main/Pytorch/torch_.py).
[torchaudio](https://github.com/aaliyanahmed1/ML-Guide/blob/main/Pytorch/torchaudio_.py).
[torchvision](https://github.com/aaliyanahmed1/ML-Guide/blob/main/Pytorch/torchvision_.py).

#### offical documentations of framekwork/refernces.
[Docs](https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html).
this documentation covers everything need to define a custom neural network\model using pytorch.
[Eplanation of neural network](https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html)
[Fine-tuning using torchvision](https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html)


### Tensorflow
Its an open-source framework for building training and deploying AI models it also provide libraries and architecures to build custom neural network/models and also some pre-trained models for inference. pre-loaded datasets post-processing and preprocessing tools for datasets handling .major libraries from tensorflow and there code implementations .

"tensorflow" core library for defining models performing operations and training models.
[tensorflow](https://github.com/aaliyanahmed1/ML-Guide/blob/main/tensorflow/tensorflow_core.py)

tensorflow_hub its model zoo of the tensorflow a great repository  for reusable pre-trained models to fine-tune and integrate them directly into the applications.  

[tensorflow_hub](https://github.com/aaliyanahmed1/ML-Guide/blob/main/tensorflow/tensorflow_hub_.py)

tf.data for loading ,preprocessing and handling dataset.
[tf.data](https://github.com/aaliyanahmed1/ML-Guide/blob/main/tensorflow/tf_data.py)

tf.image utilities for image procesing tasks.
[tf.image](https://github.com/aaliyanahmed1/ML-Guide/blob/main/tensorflow/tf_image.py)


[Obejct detection with tensorflow](https://www.tensorflow.org/hub/tutorials/tf2_object_detection)
[inference](https://github.com/aaliyanahmed1/tensorflow_/blob/main/tf2_object_detection.ipynb)

[Action recognition from video](https://www.tensorflow.org/hub/tutorials/action_recognition_with_tf_hub)
recognizing action and events from the videos using tensorflow.

This Tensorflow implementation contains deep detailed explanation of fine-tuning an object detection model on custom dataset.
all are petrain and preloaded in tensorflow no need to download them mannually .all of stuff is built-in.

[Fine-tuning Explanation](https://github.com/aaliyanahmed1/ML-Guide/blob/main/tensorflow/tensorflow_explain.py)

This was all about introduction of the framewroks now lets get back to the training .

[Training using tensorflow ](https://github.com/aaliyanahmed1/ML-Guide/blob/main/tensorflow/training.py).
This is the sample practical implementation of trainng a model on custom dataset for object detection. 



#### Fine Tuning Models for Custom detections

There are many state of teh art object detection models that can be finetune and integrate into applications .but there are some checks we need to see first before going ahead. eg License, Resources(Hardware). size and speed according to use case as we discuessed earlier.

1:**RF-DETR** by [Roboflow](https://roboflow.com/) is real-time tranformer-based object detection model ,it excels in both accuracy and speed ,fit for most of the use case  in application . its licensed under apache2.0 so it canbe used freely for commercial applciations.it has variants that varies in speed and size to fit in with the enviroment. 
| Variant        | Size (MB) | Speed (ms/image) |
| -------------- | --------- | ---------------- |
| RF‑DETR Nano   | \~15 MB   | 2.3              |
| RF‑DETR Small  | \~35 MB   | 3.5              |
| RF‑DETR base   | \~45 MB   | 4.5              |
| RF‑DETR Large  | \~300 MB  | 6.5              |

RF-DETR nano fits for integration in edge devices, mobile apps and real-time applciation where speed is crucial and low-memory is required.
[RF-DETR-nano](https://github.com/aaliyanahmed1/ML-Guide/blob/main/RF-DETR_/rfdetr-nano.py).

RF-DETR small is slightly bigger but still its fast and good fit for realtime applications and  perofroms best on GPUs.
[Rf-DETR-small](https://github.com/aaliyanahmed1/ML-Guide/blob/main/RF-DETR_/rfdetr-small.py).

RF-DETR base is ideal for server inferences for real-time application deployments.
[RF-DETR-base](https://github.com/aaliyanahmed1/ML-Guide/blob/main/RF-DETR_/rfdetrbase.py).

RF-DETR Large is  heavy-weight modle best for high accuracies on GPUs .ideal touse where accuracy is more cruicial then speed . not ideal for realtime systems.
[RF-DETR-large](https://github.com/aaliyanahmed1/ML-Guide/blob/main/RF-DETR_/rfdetrlarge.py).

For training RF-DETR on custom dataset first preprocess the dataset using ROoboflow and then downlod it accordding to the Rf-DETR format by selecting foramt in roboflow then feed it into the code and then start training bu defining which model you want to train .lets have an hands on example of it at all. [FiNE-TUNE_RF-DETR](https://github.com/aaliyanahmed1/ML-Guide/blob/main/RF-DETR_/train_rfdetr.py).


2:**YOLO** by [ultralytics](https://www.ultralytics.com/) commonly used model throughout best fit for real-time applications and fast easy to finetune .it takes image of 640x640 pixels as standard input.but its not under apache2.0 license so it cant be used freely for commercial applications.you have to pay to the company .
| Variant       | Size (MB) | Speed (ms/image) |
| ------------- | --------- | ---------------- |
| YOLO12 Nano   | \~14 MB   | 2.5              |
| YOLO12 Small  | \~27 MB   | 3.8              |
| YOLO12 Medium | \~44 MB   | 5.0              |
| YOLO12 Large  | \~89 MB   | 8.0              |


Yolo12n is ultralight and its optimized ofr edge devices dn real-time inferneces can be used in applications where spped is required and hardware is small.
[yolo12n_code](https://github.com/aaliyanahmed1/ML-Guide/blob/main/Yolo_/YOLOS_/yolo12n_.py).

Yolo12s is balanced with speed and accuracy performs well comparatively nano variant when integrate on the GPU based hardware.
[yolo12s_code](https://github.com/aaliyanahmed1/ML-Guide/blob/main/Yolo_/YOLOS_/yolo12s_.py)

Yolo12m  has significant accracy differnce from smallers ones and moderate speed ideal when deployed on server based inferneces.
[yolo12m_code](https://github.com/aaliyanahmed1/ML-Guide/blob/main/Yolo_/YOLOS_/yolo12m_.py)

Yolo12Large is high sopeed model best fot wher eprecision is crucial more then speed .mainly for medical imaging systems .
[yolo12l_code](https://github.com/aaliyanahmed1/ML-Guide/blob/main/Yolo_/YOLOS_/yolo12l_.py)

[Fine-tuning Yolo](https://github.com/aaliyanahmed1/ML-Guide/blob/main/Yolo_/YOLOS_/training.py).
these are the simple implementations of the yolo model variants for object detection tasks.

3:**FastR-CNN** by Microsoft research its a two-stage detector object detection known for its precision and high accuracy. its slightly slower then other singlestage detectors . two-stage detector means first it process ROI(region of interests) in the image and then classifies and refine bounding boxes for each region this process reduces the false positive and overlapping of objects. thats why mostly its used where speed and accuracay both are required and mainly it can be seen depl;oyed on medical imaging systems .and it variants are mainly the backbones it uses like CNNs layers(ResNet-50,ResNet-101,MobileNet). which cause differnece in speed and accuracy .

| Variant        | Backbone    | Size (MB) | Speed (ms/image) |
| -------------- | ----------- | --------- | ---------------- |
| Fast R-CNN 50  | ResNet-50   | \~120 MB  | 30               |
| Fast R-CNN 101 | ResNet-101  | \~180 MB  | 45               |
| Fast R-CNN M   | MobileNetV2 | \~60 MB   | 20               |


ResNet-50: this backbone is balanced for speed and accuracy so where both are crucial then this would be ideal fit and commonly from FastR-CNN this backbone is commonly used.
[FastR-CNN-ResNet50](https://github.com/aaliyanahmed1/ML-Guide/blob/main/FasR-CNN_/fastrcnn_resnet50.py)

ResNet-101:This has higher accuracy and slower inference so it should be integrate on precsion mandatory applications.
[FastR-CNN-ResNet101](https://github.com/aaliyanahmed1/ML-Guide/blob/main/FasR-CNN_/fastrcnn_resnet101.py).

MobileNet this variant i s again lightwieght faster but accuracy is  compromised so not so ideal .
[FastR-CNN_MobileNet](https://github.com/aaliyanahmed1/ML-Guide/blob/main/FasR-CNN_/fastrcnn_mobile.py).

These are the mostly used object detection models for commercial enterpirse applications , reserach works and medical analysis . and all of them have multiple use case centric vairants having specialization for the specific task. we have discussed them and now just we will make a list of all the possible open source object detection models that are avaiable integration in production grade applications, research and development etc . 

## Hugging face.

[Hugging Face](https://huggingface.co/) is AI platform that provide tools ,datasets and pre-trained models for Machine learning tasks.it has its wide transformer library that offer multiple ready to use open source models. its called models zoo where you can get any type of model for GenAI, Machine learning ,Computer vision and Natural language processing etc .
one of its most powerful feature is it provides inference API .which aloows to run models in cloud without setting up local enviroment .just using API for sending request and all the computation will be handled by hugging face . thre are two way to use it 1: free API good for testing and personal use and 2 is paid  plan for large applications and faster responses .
example to use hugging face APi fopr inferene.
```python
import os
from huggingface_hub import InferenceClient

client = InferenceClient(
    provider="hf-inference",
    api_key=os.environ["HF_TOKEN"],
)

output = client.image_segmentation("cats.jpg", model="facebook/mask2former-swin-base-coco-panoptic")

```

### **Transformers**
Transformers are type of deep learing architectures designed to handle sequential data using self-attention mechansims instead of traditional or convolution.They excel at capturing long-range dependencies in data. unlike older approaches that process sequences step by step ,transformers compute realtionships between all elements in  asequnce simultaneously,allowing them to capture lng-range dependencies.for our context we have to focus on Vits(vision transformers)

**Computer vision transformers** they adapt this architecture to computer vision by splitting an image into small patches, treating each patch like word in a sentence and applying same attention mechanism to learn how differnet parts of the image relate to each other . mainly used for image-to-text,text-to-image transformers fo generating captions and images .

**Vit(Vision transformer)** The first pure transformer for image classification, treating images as sequence of pathes not as pixels.
[Vits](https://huggingface.co/google/vit-base-patch16-224-in21k);Hugging-Face.


**Swin-Transformer** It uses shifted window attention mechanism for efficient scaling to high-resolution images ,it excels in segmentation,detection and calssification.
[swin](https://huggingface.co/keras-io/swin-transformers)

**BLIP/BLIP-2** A Vision language model for tasks like image captioning, VQA(visual Question nswering)and retrieval.it takes images as input and generate its caption by defining whats happeining inside the image.BLIP-2 improves teh efficiency by using pre-trained language models for better reasoning over visual inputs. pathces understanding goes to languages models and then they generate accurate caption.
[Blip](https://huggingface.co/Salesforce/blip2-flan-t5-xxl)

**Florence** Large scale vision foundation model for various  multimodel vision-language applciations. it supports takss such as image-text amtching,captioning in enterpirse and real-world production grade deployments.
[florence](https://huggingface.co/microsoft/Florence-2-base)

**Note**:These models like ViT, Swin-Transformer, BLIP/BLIP-2, and Florence are not ideal for real-time object detection on RTSP streams. They are mainly designed for high-accuracy image classification, vision-language tasks, and image captioning. These models typically require high-end GPUs with substantial memory (≥16 GB VRAM) for inference and fine-tuning, and are generally unsuitable for CPU-only or edge deployments.

### Models from Hugging Face 

**Models for Object detection with high speed:**
[Object detection models on hugging face](https://huggingface.co/models?pipeline_tag=object-detection&sort=trending)
- **YOLOv4**  
  Balanced speed and accuracy; highly optimized for real-time detection tasks.  
  **Speed:** ~65 FPS (V100)  
  **Accuracy:** ~43.5% AP (COCO-dataset)
  [Yolov4Tiny](https://huggingface.co/gbahlnxp/yolov4tiny)

  **Yolos-Tiny**
   
  [yolos-tiny](https://huggingface.co/hustvl/yolos-tiny)
- **YOLOv7**  
  State-of-the-art real-time detection model with top-tier accuracy.  
  **Speed:** 30–160 FPS  
  **Accuracy:** ~56.8% AP (30+ FPS)
   [Yolov7](https://huggingface.co/kadirnar/yolov7-tiny-v0.1)

- **SSD (Single-Shot Detector)**  
  Lightweight single-stage detector suitable for real-time applications.  
  **Speed:** ~58 FPS  
  **Accuracy:** ~72.1% (Pascal VOC)

- **EfficientDet (D0–D7)**  
  Scalable and efficient detectors with excellent COCO performance.  
  **Speed:** 30–50 FPS (varies by variant)  
  **Accuracy:** Up to ~55.1% AP (COCO)
  [EfficientNet](https://huggingface.co/google/efficientnet-b7).

- **RetinaNet**  
  One-stage detector with Focal Loss to handle class imbalance effectively.  
  **Speed:** ~30 FPS  
  **Accuracy:** High
[RetinaNet](https://huggingface.co/keras-io/Object-Detection-RetinaNet).

- **RT-DETR (R50)**  
  Real-Time DETR optimized for fast inference.  
  **Speed:** 35 Fps 
  **Accuracy:** Good overall performance
[RT-DETR](https://huggingface.co/PekingU/rtdetr_r101vd_coco_o365).

### Deployment
Deployment is very typical part of every Machine learning workflow.when it comes to deployment maintaining fps for real-time systems becoems nightmare of MLOps architects so thats why the universal way to deploy model and maintain performance is to decouple it from training framework, that simplifis and reduces down burden of heavy dependencies and speed up the process is exporting model in ONNX(Open Neural Network Exchange) format.this simplifies integration of model and makes it compatible.

#### **ONNX** 
 Its is an open standard format for representing machine learning models. Exporting models to ONNX decouples them from the original training framework, making them easier to integrate into different platforms, whether on a server, multiple edge devices, or in the cloud. It ensures compatibility across various tools and allows optimized inference on different hardware setups, helping maintain real-time performance.
here is simple minimal code implementation to export model that is in its training framework to export it ot ONNX format for cross-platofrm integration and further optimizations.

```python

"""RF-DETR to ONNX conversion script."""

import torch
from rfdetr import RFDETRSmall


def convert_rfdetr_to_onnx():
    """Convert RF-DETR model to ONNX format."""
    # Load pretrained RF-DETR model
    model = RFDETRSmall()
    model.eval()

    # Create dummy input tensor
    dummy_input = torch.randn(1, 3, 800, 800)

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        "rfdetr_small.onnx",
        input_names=["input"],
        output_names=["logits", "boxes"],
        dynamic_axes={
            "input": {0: "batch", 2: "height", 3: "width"},
            "logits": {0: "batch"},
            "boxes": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
        verbose=True,
    )

    print("✅ RF-DETR Small exported to rfdetr_small.onnx")


if __name__ == "__main__":
    convert_rfdetr_to_onnx()
    
```

#### **Onxruntime**
 is a high-performance inference engine designed to run ONNX modles efficiently accross different platforms.it takes the ONNX model and applies grapgh optimization,operator fusion and quantizations to reduce memory usage and computation time .so models run faster on servers,cloud enviroments and on multiple edge devices without needing original training framework.it can also speedup training process of large models by just making simple changes in code it can make training faster and effcient without changing workflow too much.  


[Onnxruntime Docs](https://onnxruntime.ai/docs/get-started/with-python.html#install-onnx-runtime)

[Onnxruntime for training](https://onnxruntime.ai/training).


**Graph Optimization:** It rearranges and simplifies the model's performance computation graph to remov unnecassary steps, making it run faster.like combining adjacent layers or removing unused nodes.these all optimizations are automatically applied in ONNx Runtime when session is created with grapgh optimization enabled(oRT_ENABLE_ALL).reduing memory usage and CPU/GPU cycles,helping maintain higher FPS for real time inference 

```python
import onnxruntime as ort

# ---------------------------
# Create session options for ONNX Runtime
# ---------------------------
session_options = ort.SessionOptions()

# ---------------------------
# Set graph optimization level
# ---------------------------
# ORT_ENABLE_BASIC: Applies basic, safe optimizations like removing unused nodes
# ORT_ENABLE_EXTENDED: Includes more optimizations such as some operator fusions
# ORT_ENABLE_ALL: Applies all available optimizations including aggressive fusions and node eliminations
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

# ---------------------------
# Load ONNX model with the chosen optimization level
# ---------------------------
session = ort.InferenceSession("model.onnx", sess_options=session_options)

# ---------------------------
# ✅ Explanation:
# 1. Creating SessionOptions allows you to configure how the ONNX model runs.
# 2. Graph optimization automatically rearranges computations, removes unnecessary nodes,
#    and may combine adjacent operations for faster inference.
# 3. ORT_ENABLE_ALL is recommended for real-time systems to maximize FPS.
# ---------------------------
print("✅ ONNX model loaded with graph optimization enabled!")
```

**Operator Fusion:** This function merges multiple small operations into a single, more efficient operation to reduce processing overhead. 
fusing cnv+ BtachNorm + ReLU into one step.it reduces number of kernels launches on CPU\GPU.that spped up inference and lowers memory overhead. it happens automatically when grapgh optimzation is enable=True.

```python
import onnx
from onnxruntime.transformers.optimizer import optimize_model

# ---------------------------
# Step 1: Define paths for models
# ---------------------------
onnx_model_path = "model.onnx"              # Original ONNX model
fused_model_path = "model_fused.onnx"       # Path to save operator-fused model

# ---------------------------
# Step 2: Apply Operator Fusion
# ---------------------------
# Operator fusion merges multiple operations into a single, more efficient operation
# Example: Conv + BatchNorm + ReLU → single fused operator
# This reduces kernel launches on GPU/CPU and improves inference speed
# The 'model_type' parameter can be adjusted if using a transformer-like model.
# For general vision models like RF-DETR, using "rf-detr" or "generic" is appropriate.
fused_model = optimize_model(onnx_model_path, model_type="rf-detr")

# ---------------------------
# Step 3: Save the fused model
# ---------------------------
fused_model.save_model_to_file(fused_model_path)

# ---------------------------
# ✅ Explanation:
# 1. Operator Fusion improves runtime by combining adjacent operations.
# 2. It reduces computation overhead and memory usage.
# 3. Works together with graph optimization for maximum inference speed.
# ---------------------------
print(f"✅ Operator fusion applied! Fused model saved at {fused_model_path}")
```

**Quantization:** It converts high-precision numbers(floating point32) into lower precision(INT8) to reduce memory and improve speed with minimal accuracy loss. compressing weights from 32 floating point to 8-bit integers.Quantization includes several advanced techniques beong simple dyunamuc int8 quantization .lets discuss them one-by-one .

**1: Dynamic Quantization** It Converts weights only to INT* during runtime .activation functions ( sigmoid,Relu etc) remains same FP#@.its fast adn easy and no need for training again and again for updates like that. ideal where all of the workflow is deployed on CPU .
```python
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

# ---------------------------
# Step 1: Define paths
# ---------------------------
onnx_model_path = "model.onnx"          # Original FP32 ONNX model
quantized_model_path = "model_int8.onnx" # Path to save INT8 quantized model

# ---------------------------
# Step 2: Apply Dynamic Quantization
# ---------------------------
# Converts weights (FP32 → INT8) for selected ops (e.g., MatMul, GEMM, Attention)
quantize_dynamic(
    model_input=onnx_model_path,
    model_output=quantized_model_path,
    weight_type=QuantType.QInt8  # Quantize weights to INT8
)

# ---------------------------
# Step 3: ✅ Explanation
# ---------------------------
# 1. Reduces model size (weights now INT8).
# 2. Faster CPU inference (uses INT8 optimized kernels in ONNX Runtime).
# 3. No retraining required – works directly on exported ONNX model.
# ---------------------------
print(f"✅ Dynamic Quantization applied! Quantized model saved at {quantized_model_path}")

```

**2:Mixed Precision/Fp16 Quantization** this reduces precision from FP32 to FP16 often used on hardware with GPUs to speed inference while keeping accuracy close to full precision yet speeding up the process.

```python
import onnx
from onnxconverter_common import float16

# ---------------------------
# Step 1: Define paths
# ---------------------------
onnx_model_path = "rfdetr_small.onnx"          # Original FP32 ONNX model
fp16_model_path = "rfdetr_small_fp16.onnx"    # Path to save FP16 model

# ---------------------------
# Step 2: Load the ONNX model
# ---------------------------
model = onnx.load(onnx_model_path)

# ---------------------------
# Step 3: Convert model weights to FP16
# ---------------------------
# float16 conversion reduces memory usage and speeds up GPU inference
fp16_model = float16.convert_float_to_float16(model)

# ---------------------------
# Step 4: Save the FP16 ONNX model
# ---------------------------
onnx.save(fp16_model, fp16_model_path)

# ---------------------------
# ✅ Explanation:
# 1. Converts FP32 weights to FP16 for GPU acceleration.
# 2. Reduces model memory footprint by ~50%.
# 3. Maintains accuracy close to FP32 (slight precision loss possible).
# 4. Best used with GPU runtime; CPUs do not benefit much.
# ---------------------------
print(f"✅ FP16 Mixed Precision applied! Model saved at {fp16_model_path}")

```

**3:Pruning + Quantization** pruning removes unimportant weoghts usually small-magnitude from th network reduces models size and computations which increease FPS in real-time.

```python
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime.tools import onnx_pruning

# ---------------------------
# Step 1: Define paths
# ---------------------------
onnx_model_path = "rfdetr_small.onnx"          # existing ONNX model
pruned_model_path = "rfdetr_small_pruned.onnx" # Path for pruned model
quantized_model_path = "rfdetr_small_pruned_int8.onnx"  # Path for INT8 quantized model

# ---------------------------
# Step 2: Prune unimportant weights
# ---------------------------
# Using ONNX Runtime pruning tool (magnitude-based)
# Removes low-magnitude weights to reduce model size
pruned_model = onnx_pruning.prune_model(
    model_path=onnx_model_path,
    prune_ratio=0.2,           # Remove 20% of small-magnitude weights
    output_model_path=pruned_model_path
)

# ---------------------------
# Step 3: Apply dynamic INT8 quantization
# ---------------------------
quantize_dynamic(
    model_input=pruned_model_path,
    model_output=quantized_model_path,
    weight_type=QuantType.QInt8
)

# ---------------------------
# ✅ Explanation:
# 1. Pruning reduces unnecessary weights → fewer computations, faster inference.
# 2. Dynamic INT8 quantization reduces model size and speeds up CPU inference.
# 3. The final ONNX model is optimized for real-time deployment.
# ---------------------------
print(f"✅ Pruned + Quantized ONNX model saved at {quantized_model_path}")
```

These optimization techniques are applied during inference to reduce model;s load on memory and remove extra operations and combine small essential operations to make model optimized task centric.this make model more suitable for the operation and reduce computational cost allowing the model to run faster, maintain higher FPS, and respond quickly to incoming data in real-time applications.
