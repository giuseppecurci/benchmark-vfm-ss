import torch 
from torchmetrics import JaccardIndex
import torch.nn.functional as F
import numpy as np 
import torch
from torch.utils.data import DataLoader
from models.linear_decoder import LinearDecoder
from bravo_datasets import CityscapesDataset, ACDCDataset
import argparse

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

def parse_args():
    parser = argparse.ArgumentParser(description="Parse model and dataset arguments.")
    
    parser.add_argument("--model.network.encoder_name", type=str, required=True,
                        help="Name of the encoder for the model.")
    parser.add_argument("--model.network.patch_size", type=int, required=True,
                        help="Patch size for the network.")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to the model checkpoint.")
    parser.add_argument("--img_size", type=int, required=True,
                        help="Image size as two integers: height width.")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name or path.")
    parser.add_argument("--root", type=str, required=True,
                        help="Root directory of the dataset.")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size for evaluation.")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of workers for data loader.")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    model = LinearDecoder(
        encoder_name=getattr(args, "model.network.encoder_name"),
        num_classes = 19,
        img_size = (args.img_size,args.img_size),
        sub_norm = False, 
        patch_size = getattr(args, "model.network.patch_size"),
        pretrained = True
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    checkpoint = torch.load(args.checkpoint_path, map_location=device) 
    model.load_state_dict(checkpoint['state_dict'])
    if args.dataset in ['synrain', 'out_of_context', 'synflare']:
        dataset = CityscapesDataset(root_dir=args.root, dataset=args.dataset, img_size=(args.img_size, args.img_size))
    elif args.dataset == 'ACDC':
        dataset = ACDCDataset(root_dir=args.root, split='val', img_size=(args.img_size, args.img_size))
    else:
        raise ValueError("Invalid dataset name. Choose one of: 'synrain', 'out_of_context', 'synflare' or 'ACDC'.")
    eval_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    normal_eval(model=model, 
                img_size=(args.img_size, args.img_size), 
                device=device, 
                num_classes=19, 
                eval_dataloader=eval_dataloader)