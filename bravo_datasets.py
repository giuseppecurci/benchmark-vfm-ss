import os
from torch.utils.data import Dataset
import cv2
import glob 
import torch 
import numpy as np
import torchvision

class CityscapesDataset(Dataset):
    def __init__(self, root_dir, dataset, img_size=(1024, 1024)):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, "leftImg8bit")
        self.label_dir = os.path.join(root_dir, "gtFine")
        self.img_size = img_size
        # Find all image paths
        self.image_paths = sorted(glob.glob(os.path.join(self.image_dir, "*", "*_leftImg8bit.png")))
        # Ignore 'road', 'pavement' and 'terrain' classes if using out_of_context dataset
        if dataset not in ["synrain", "out_of_context", "synflare"]:
            raise ValueError("Invalid dataset name. Choose one of: 'synrain', 'out_of_context' or 'synflare'.")
        self.ignore_idx = [0,1,9] if dataset == "out_of_context" else []  
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
        img_path, label_path = self.pairs[idx]

        # Load image (BGR format)
        image = cv2.imread(img_path)  # No normalization, No conversion to RGB
        image = image.astype(np.float32)  # Convert to float32 before dividing
        image /= 255.0  # Normalize image to [0, 1]
        
        # Load label (Grayscale with integer values)
        label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)

        # Resize the label and image to match the training image size
        label = cv2.resize(label, (self.img_size[0], self.img_size[1]), interpolation=cv2.INTER_NEAREST)
        image = cv2.resize(image, (self.img_size[0], self.img_size[1]), interpolation=cv2.INTER_LINEAR)
            
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
    
class ACDCDataset(Dataset):
    def __init__(self, root_dir, split='train', img_size=(1024,1024)):
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.data = []
        self.cs_mappings = self.get_cityscapes_mapping()
        self._load_data()
    
    def _load_data(self):
        weather_conditions = ['fog', 'night', 'rain', 'snow']
        for weather in weather_conditions:
            rgb_dir = os.path.join(self.root_dir, 'rgb_anon', weather , self.split)
            gt_dir = os.path.join(self.root_dir, 'gt', weather, self.split)
            
            if not os.path.exists(rgb_dir) or not os.path.exists(gt_dir):
                continue
            
            for sequence in os.listdir(rgb_dir):
                rgb_seq_path = os.path.join(rgb_dir, sequence)
                gt_seq_path = os.path.join(gt_dir, sequence)
                
                if not os.path.isdir(rgb_seq_path) or not os.path.isdir(gt_seq_path):
                    continue
                
                for file in os.listdir(rgb_seq_path):
                    if file.endswith('_rgb_anon.png'):
                        base_name = file.replace('_rgb_anon.png', '')
                        rgb_path = os.path.join(rgb_seq_path, file)
                        gt_inv_path = os.path.join(gt_seq_path, base_name + '_gt_invIds.png')
                        gt_label_path = os.path.join(gt_seq_path, base_name + '_gt_labelIds.png')
                        
                        if os.path.exists(gt_inv_path) and os.path.exists(gt_label_path):
                            self.data.append((rgb_path, gt_inv_path, gt_label_path))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        rgb_path, gt_inv_path, gt_label_path = self.data[idx]
        
        rgb_img = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB) / 255.0
        rgb_img = cv2.resize(rgb_img, (self.img_size[0], self.img_size[1]))
        
        gt_inv = cv2.imread(gt_inv_path, cv2.IMREAD_GRAYSCALE)
        gt_inv = cv2.resize(gt_inv, (self.img_size[0], self.img_size[1]), interpolation=cv2.INTER_NEAREST)
        
        gt_label = cv2.imread(gt_label_path, cv2.IMREAD_GRAYSCALE)
        gt_label = cv2.resize(gt_label, (self.img_size[0], self.img_size[1]), interpolation=cv2.INTER_NEAREST)
        
        gt_label[gt_inv == 1] = 255  # Set invalid pixels to 255
        for k, v in self.cs_mappings.items():
            gt_label[gt_label == k] = v

        rgb_img = torch.tensor(rgb_img, dtype=torch.float32).permute(2, 0, 1)  # HWC -> CHW
        gt_label = torch.tensor(gt_label, dtype=torch.long)
        
        return rgb_img, gt_label
    
    @staticmethod
    def get_cityscapes_mapping():
        return {
            class_.id: class_.train_id
            for class_ in torchvision.datasets.Cityscapes.classes
        } 