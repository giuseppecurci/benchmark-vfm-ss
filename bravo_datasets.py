import os
from torch.utils.data import Dataset
import cv2
import glob 
import torch 
import numpy as np
import torchvision

class CityscapesDataset(Dataset):
    def __init__(self, root_dir, img_size=(1024, 1024), ignore_idx=[]):
        """
        Args:
            root_dir (str): Path to the root dataset directory (e.g., /media/data/workspace_Giuseppe/code/datasets/BRAVO/synrain)
        """
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, "leftImg8bit")
        self.label_dir = os.path.join(root_dir, "gtFine")
        self.img_size = img_size
        # Find all image paths
        self.image_paths = sorted(glob.glob(os.path.join(self.image_dir, "*", "*_leftImg8bit.png")))
        self.ignore_idx = ignore_idx   
        # Match them with corresponding ground truth files
        self.pairs = []
        for img_path in self.image_paths:
            city = os.path.basename(os.path.dirname(img_path))  # Get city name
            filename = os.path.basename(img_path).replace("_leftImg8bit.png", "")

            # Ground truth path
            label_path = os.path.join(self.label_dir, city, f"{filename}_gtFine_labelIds.png")

            # Only keep valid pairs
            if os.path.exists(label_path):
                self.pairs.append((img_path, label_path))

        self.cs_mappings = self.get_cityscapes_mapping()

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        # NOTE: We are not normalizing the images here because 
        # they're normalized in the model's forward pass
        
        img_path, label_path = self.pairs[idx]

        # Load image (BGR format)
        image = cv2.imread(img_path)  # No normalization, No conversion to RGB
        image = image.astype(np.float32)  # Convert to float32 before dividing
        image /= 255.0  # Normalize image to [0, 1]
        
        # Load label (Grayscale with integer values)
        label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)

        # Resize the label and image to match the training image size
        label = cv2.resize(label, (self.img_size[1], self.img_size[1]), interpolation=cv2.INTER_NEAREST)
        image = cv2.resize(image, (self.img_size[1], self.img_size[1]), interpolation=cv2.INTER_LINEAR)
            
        # Convert to torch tensors
        image = torch.tensor(image).permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
        label = torch.tensor(label, dtype=torch.long)  # Keep original integer IDs

        for k, v in self.cs_mappings.items():
            label[label == k] = v
        for i in self.ignore_idx:
            label[label == i] = 255
            
        return image, label
    
    @staticmethod
    def get_cityscapes_mapping():
        return {
            class_.id: class_.train_id
            for class_ in torchvision.datasets.Cityscapes.classes
        } 