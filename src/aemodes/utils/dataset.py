import numpy as np
import pickle

import tifffile as tif
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

import lightning as L

import pycocotools.mask as mask_utils

class ShotDataset(Dataset):
    def __init__(self, shots, X, y):
        self.shots = shots
        self.X = X
        self.y = y
        
        lenshot = 3905
        self.nwin = 11
        self.lenwin = lenshot // self.nwin
        self.hoplen = lenshot // self.nwin
    
    def __len__(self):
        return len(self.shots) * self.nwin
    
    def __getitem__(self, idx):
        shot_idx, win_idx = idx // self.nwin, idx % self.nwin
        start_idx = win_idx * self.hoplen
        end_idx = start_idx + self.lenwin
        X = self.X[shot_idx]
        X = torch.tensor(np.stack([X['r0'],X['v1'],X['v2'],X['v3']])[:,start_idx:end_idx])
        y = torch.tensor(self.y[shot_idx][start_idx:end_idx])
        X = X.transpose(1, 2)
        y = y.transpose(0, 1)
        return {
            'shot': self.shots[shot_idx],
            'X': X.float(),
            'y': y.float(),
        }

def coco_collate_fn(batch):
    return tuple(zip(*batch))

class COCODataset(Dataset):
    
    def __init__(self, root_dir, mode='train', transforms=None):
        self.root_dir = Path(root_dir)
        self.mode = mode
        self.transforms = transforms
        
        # Load annotations
        ann_file = self.root_dir / f'annotations_{mode}.json'
        with open(ann_file, 'r') as f:
            self.coco_data = json.load(f)
        
        self.images = self.coco_data['images']
        
        # Build annotation lookup
        self.annotations = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.annotations:
                self.annotations[img_id] = []
            self.annotations[img_id].append(ann)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_id = img_info['id']
        height = img_info['height']
        width = img_info['width']
        
        # Load image
        img_path = self.root_dir / self.mode / img_info['file_name']
        image = tif.imread(img_path)

        # Load annotations
        annotations = self.annotations.get(img_id, [])
        
        boxes, labels, areas, masks = [], [], [], []
        for ann in annotations:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])
            areas.append(ann['area'])

            if isinstance(ann['segmentation'], dict):
                rle = ann['segmentation']
                mask = mask_utils.decode(rle)
                masks.append(mask)
            else:
                mask = np.zeros((height, width), dtype=np.uint8)
                mask[int(y):int(y+h), int(x):int(x+w)] = 1
                masks.append(mask)
        
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = torch.zeros((0, height, width), dtype=torch.uint8)
            areas = torch.zeros((0,), dtype=torch.float32)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            masks = torch.as_tensor(np.stack(masks), dtype=torch.uint8)
            areas = torch.as_tensor(areas, dtype=torch.float32)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
            'image_id': torch.tensor([img_id]),
            'area': areas,
            'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64),
        }
        
        if self.transforms:
            image = self.transforms(image)
        
        return image, target

def load_dataset(datapath):
    [train_shots, X_train,y_train,valid_shots, X_valid,y_valid] = pickle.load(open(datapath,'rb'))

    train_dataset = ShotDataset(train_shots, X_train, y_train)
    valid_dataset = ShotDataset(valid_shots, X_valid, y_valid)

    return train_dataset, valid_dataset

def coco_transforms(image):
    image = image.astype(np.float32)
    image = np.stack([image] * 3, axis=0)
    image = torch.from_numpy(image)
    return image

class COCODataModule(L.LightningDataModule):
    
    def __init__(
        self,
        data_path,
        batch_size=12,
        num_workers=1,
        transforms=coco_transforms,
    ):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transforms = transforms
    
    def setup(self, stage=None):
        if stage == 'fit':
            self.train_dataset = COCODataset(
                self.data_path,
                mode='train',
                transforms=self.transforms,
            )
            self.valid_dataset = COCODataset(
                self.data_path,
                mode='valid',
                transforms=self.transforms,
            )
        if stage == 'test':
            self.test_dataset = COCODataset(
                self.data_path,
                mode='valid',
                transforms=self.transforms,
            )
        if stage == 'predict':
            self.predict_dataset = COCODataset(
                self.data_path,
                mode='valid',
                transforms=self.transforms,
            )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
            collate_fn=coco_collate_fn,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            collate_fn=coco_collate_fn,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            collate_fn=coco_collate_fn,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            collate_fn=coco_collate_fn,
        )