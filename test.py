from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torchvision.transforms as T
import torch
import torch.nn as nn

weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = maskrcnn_resnet50_fpn_v2(weights=weights)

# Replace the box predictor
in_features = model.roi_heads.box_predictor.cls_score.in_features  # type: ignore[union-attr]
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)

# Replace the mask predictor
in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels  # type: ignore[union-attr]
hidden_layer = 256
model.roi_heads.mask_predictor = MaskRCNNPredictor(
    in_features_mask,
    hidden_layer,
    2,
)

# Create a custom transform with just resize
from torchvision.models.detection.transform import GeneralizedRCNNTransform

# Create transform with only resize (no normalization)
custom_transform = GeneralizedRCNNTransform(
    min_size=800,
    max_size=1333,
    image_mean=[0.0, 0.0, 0.0],  # No normalization
    image_std=[1.0, 1.0, 1.0],   # No normalization
)

# Replace the model's transform
model.transform = custom_transform

print(model)
# model.eval()

# # Mask R-CNN expects a list of images and returns predictions
# with torch.no_grad():
#     x = torch.randn(3, 224, 224)
#     # Model expects a list of tensors
#     predictions = model([x])

# print('success')