import torch 
from torchmetrics import JaccardIndex
import torch.nn.functional as F
import numpy as np 

def normal_eval(
        model,
        img_size,
        device,
        eval_dataloader,
        ignore_index=255, 
        num_classes=19,  
        ):

    with torch.no_grad():
        iou_scores = []
        iou_metric = JaccardIndex(task="multiclass", num_classes=num_classes, ignore_index=ignore_index)
        for i, (image, target) in enumerate(eval_dataloader):
            image = image.to(device)  
            target = target.to(device)  
            iou_metric = iou_metric.to(image.device)

            # Forward pass
            logits = model(image)  # Model output: (B, num_classes, H, W)

            # Resize logits to match target size
            logits = F.interpolate(logits, size=img_size, mode="bilinear", align_corners=False)
            
            # Convert logits to class predictions
            predicted = torch.argmax(logits, dim=1)  # Shape: (B, H, W)

            # Compute IoU for each image in the batch
            batch_iou = iou_metric(predicted.flatten(), target.flatten()).item()

            # Store IoU scores
            iou_scores.append(batch_iou)

            print(f"Processed {i+1}/{len(eval_dataloader)} batches - Mean IoU: {np.nanmean(batch_iou):.4f}")

    # Compute final mean IoU
    mean_iou = np.mean(iou_scores)
    print(f"\nFinal Mean IoU: {mean_iou:.4f}")

def main():
    pass 