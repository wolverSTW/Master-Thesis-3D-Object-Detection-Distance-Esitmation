import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
import kagglehub
import logging

logger = logging.getLogger(__name__)

class Kitti3DDataset(Dataset):
    def __init__(self, data_dir: str = "./data", split: str = "training"):
        self.data_dir = data_dir
        self.split = split
        self.image_dir = os.path.join(data_dir, split, "image_2")
        self.label_dir = os.path.join(data_dir, split, "label_2")
        self.calib_dir = os.path.join(data_dir, split, "calib")
        
        # Ingestion Automation Trigger
        if not os.path.exists(self.image_dir):
            logger.info("📡 Auto-downloading Dataset via Kagglehub...")
            kagglehub.dataset_download("klemenko/kitti-dataset")
            
        self.file_ids = sorted([os.path.splitext(f)[0] for f in os.listdir(self.image_dir) if f.endswith('.png')]) if os.path.exists(self.image_dir) else ["000000"]

    def __len__(self) -> int:
        return len(self.file_ids)

    def __getitem__(self, idx: int) -> dict:
        file_id = self.file_ids[idx]
        img_path = os.path.join(self.image_dir, f"{file_id}.png")
        image = cv2.imread(img_path) if os.path.exists(img_path) else np.zeros((375, 1242, 3), dtype=np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        calib_path = os.path.join(self.calib_dir, f"{file_id}.txt")
        P2 = np.eye(4)[:3, :]
        if os.path.exists(calib_path):
            with open(calib_path, 'r') as f:
                for line in f:
                    if line.startswith('P2:'):
                        P2 = np.fromstring(line.split(':', 1)[1].strip(), sep=' ').reshape(3, 4)
                        break
        
        label_path = os.path.join(self.label_dir, f"{file_id}.txt")
        gt_boxes3d = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        if os.path.exists(label_path):
            parsed_boxes = []
            with open(label_path, 'r') as f:
                for line in f:
                    data = line.strip().split(' ')
                    if data[0] in ['Car', 'Pedestrian', 'Cyclist']:
                        parsed_boxes.append([float(data[8]), float(data[9]), float(data[10]), float(data[11]), float(data[12]), float(data[13]), float(data[14])])
            if parsed_boxes: gt_boxes3d = parsed_boxes

        return {"image": image_tensor, "calib_P2": torch.from_numpy(P2).float(), "gt_boxes3d": torch.tensor(gt_boxes3d, dtype=torch.float32)}