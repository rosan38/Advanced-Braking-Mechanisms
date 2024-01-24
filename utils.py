import json
import os

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class PedestrianDataset(Dataset):
    # Custom Dataset class for pedestrian detection.
    def __init__(self, images, annotations, transform=None):
        self.transform = transform
        self.imgs = images
        self.annotations = annotations

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # Load and transform the image and annotations
        img_path = self.imgs[idx]
        ann_path = self.annotations[idx]
        image = Image.open(img_path).convert("RGB")

        with open(ann_path) as f:
            annotations = json.load(f)

        boxes = self.extract_boxes(annotations)
        labels, area, iscrowd = self.generate_labels(boxes)
        image_id = torch.tensor([idx])

        target = {
            "boxes": boxes, "labels": labels, "image_id": image_id,
            "area": area, "iscrowd": iscrowd
        }

        if self.transform:
            image = self.transform(image)
        return image, target

    @staticmethod
    def extract_boxes(annotations):
        # Extract bounding boxes from annotations
        boxes = []
        for ann in annotations:
            if ann['lbl'] in ['person', 'people']:
                x, y, width, height = ann['pos']
                boxes.append([x, y, x + width, y + height])
        return torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)

    @staticmethod
    def generate_labels(boxes):
        # Generate labels, area, and iscrowd for each box
        num_boxes = len(boxes)
        labels = torch.ones((num_boxes,), dtype=torch.int64)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_boxes,), dtype=torch.int64)
        return labels, area, iscrowd


def collate_fn(batch):
    # Custom collate function for DataLoader
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    images = torch.stack(images, 0)
    return images, targets


def get_transform():
    transforms = [T.ToTensor()]
    return T.Compose(transforms)


def get_model(num_classes):
    # Load and modify a pre-trained Faster R-CNN model
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def load_all_data(root, sets):
    # Load all images and annotations from the given sets.
    all_images = []
    all_annotations = []

    for set_name in sets:
        img_dir = os.path.join(root, set_name, "images")
        ann_dir = os.path.join(root, set_name, "annotations")

        for img_file in sorted(os.listdir(img_dir)):
            if img_file.endswith('.jpg'):
                all_images.append(os.path.join(img_dir, img_file))
                ann_file = img_file.replace('.jpg', '.json')
                ann_path = os.path.join(ann_dir, ann_file)
                if os.path.isfile(ann_path):
                    all_annotations.append(ann_path)

    return all_images, all_annotations
