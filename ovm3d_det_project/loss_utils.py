import torch
import torch.nn as nn

class YOLO3DLoss(nn.Module):
    def __init__(self):
        super(YOLO3DLoss, self).__init__()
        self.l1_loss = nn.SmoothL1Loss(reduction='mean')
        self.mse_loss = nn.MSELoss(reduction='mean')

    def forward(self, preds: dict, targets: dict) -> dict:
        loss_2d = self.l1_loss(preds["pred_2d"][:, :4], targets["gt_2d"][:, :4]) + self.mse_loss(preds["pred_2d"][:, 4], targets["gt_2d"][:, 4])
        loss_dim = self.l1_loss(preds["pred_3d"][:, :3], targets["gt_dim"])
        loss_angle = self.mse_loss(preds["pred_3d"][:, 6:8], targets["gt_angle"])
        return {"total_loss": loss_2d + loss_dim + (0.5 * loss_angle), "loss_2d": loss_2d, "loss_dim": loss_dim, "loss_angle": loss_angle}