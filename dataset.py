import torch 
import torch.nn as nn
from torch.utils.data import Dataset
import json
import PIL
import os 
import torchvision.transforms as T
import torchvision 
from PIL import Image
from utils import box_cxcywh_to_xyxy

class ObjectDetectionDataset(Dataset):
    def __init__(self, data_dir, annotation_file, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.annotations = json.load(open(annotation_file, 'r'))
        self.image_ids = list(self.annotations['images'])
        self.categories = {category['id']: category['name'] for category in self.annotations['categories']}
        self.category_mapping = {cat_id: idx for idx, cat_id in enumerate(self.categories.keys())}
        self.num_classes = len(self.categories)

        if self.transform is None:
            self.transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_info = self.annotations['images'][idx]
        image_id = image_info['id']
        image_path = os.path.join(self.data_dir, image_info['file_name'])
        image = Image.open(image_path).convert('RGB')

        w, h = image.width, image.height
        annotations = [ann for ann in self.annotations['annotations'] if ann['image_id'] == image_id]
        boxes = [ann['bbox'] for ann in annotations]
        labels = [self.category_mapping[ann['category_id']] for ann in annotations]

        # convert from [x1, y1, w, h] to [x1, y1, x2, y2]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        # convert to (cx, cy, w, h) format and normalize to image size 
        # since this is what detr loss optimizes against
        x0, y0, x1, y1 = boxes.unbind(-1)
        b = [(x0 + x1) / 2, (y0 + y1) / 2,
            (x1 - x0), (y1 - y0)]
        boxes = torch.stack(b, dim=-1)
        # normalize to image size in the range [0, 1]
        boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)

        target = {
            'boxes': boxes,
            'labels': torch.tensor(labels, dtype=torch.int64),
            'size': torch.tensor([w, h], dtype=torch.int64),
            'original_image': image,
        }

        if self.transform:
            image = self.transform(image)

        return image, target
