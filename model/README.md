---
license: agpl-3.0
library: ultralytics
tags:
- object-detection
- pytorch
- roboflow-universe
- pickle
- face-detection
---
# Face Detection using YOLOv8

This model was fine tuned on a dataset of over 10k images containing human faces. The model was fine tuned for 100 epochs with a batch size of 16 on a single NVIDIA V100 16GB GPU, it took around 140 minutes for the fine tuning to complete.

## Downstream Tasks

- __Face Detection__: This model can directly use this model for face detection or it can be further fine tuned own a custom dataset to improve the prediction capabilities.
- __Face Recognition__: This model can be fine tuned to for face recognition tasks as well, create a dataset with the images of faces and label them accordingly using name or any ID and then use this model as a base model for fine tuning.

# Example Usage

```python
# load libraries
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from supervision import Detections
from PIL import Image

# download model
model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")

# load model
model = YOLO(model_path)

# inference
image_path = "/path/to/image"
output = model(Image.open(image_path))
results = Detections.from_ultralytics(output[0])

```


# Links

- __Dataset Source__: [Roboflow Universe](https://universe.roboflow.com/large-benchmark-datasets/wider-face-ndtcz/dataset/1)
- __Weights & Biases__: [Run Details](https://wandb.ai/2wb2ndur/Face-Detection/overview?workspace=user-2wb2ndur) 
- __Training Artifacts__: [training-artifacts](./fine-tune-artifacts/)