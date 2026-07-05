import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
import time

# စောစောက ဆောက်ခဲ့သော သီးသန့် Modules များကို လှမ်း၍ Import ယူခြင်း
from data_utils import Kitti3DDataset
from model_utils import YOLO3DNetwork
from loss_utils import YOLO3DLoss

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("🎬 Launching Production-Grade CLI Training Pipeline...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = Kitti3DDataset()
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
    
    model = YOLO3DNetwork().to(device)
    criterion = YOLO3DLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    for epoch in range(1, 3):
        start = time.time()
        model.train()
        for batch in dataloader:
            images = batch["image"].to(device)
            targets = {
                "gt_2d": torch.cat([batch["gt_boxes3d"][:, 0, 3:7], batch["gt_boxes3d"][:, 0, 0:1]], dim=1).to(device),
                "gt_dim": batch["gt_boxes3d"][:, 0, 0:3].to(device),
                "gt_angle": torch.stack([torch.sin(batch["gt_boxes3d"][:, 0, 6]), torch.cos(batch["gt_boxes3d"][:, 0, 6])], dim=1).to(device)
            }
            optimizer.zero_grad()
            preds = model(images)
            losses = criterion(preds, targets)
            losses["total_loss"].backward()
            optimizer.step()
            
        logger.info(f"🏆 Epoch [{epoch}/2] Completed in {time.time()-start:.2f}s | Total Loss: {losses['total_loss'].item():.4f}")

if __name__ == "__main__":
    main()