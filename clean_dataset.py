import os, glob 
import matplotlib.pyplot as plt

def clean_out_of_context(root_dir):
    '''
    Remove all mask files that are not _gtFine_labelIds.png or do not have a corresponding image file.
    Given the design of the Cityscapes dataset, this function is not strictly necessary.
    '''
    folders = ['leftImg8bit', 'gtFine']
    image_path = os.path.join(root_dir, folders[0])
    mask_path = os.path.join(root_dir, folders[1])
    cities = [city for city in os.listdir(image_path) if os.path.isdir(os.path.join(image_path, city))]

    for city in cities:
        city_image_folder = os.path.join(image_path, city)
        city_mask_folder = os.path.join(mask_path, city)
        
        if not os.path.exists(city_mask_folder):
            raise ValueError('The path does not exist')  # Skip if the mask folder doesn't exist (unlikely but safe)
        
        # Get all valid image filenames (without suffix)
        valid_images = set()
        for img_file in glob.glob(os.path.join(city_image_folder, "*_leftImg8bit.png")):
            base_name = os.path.basename(img_file).replace("_leftImg8bit.png", "")
            valid_images.add(base_name)
        
        # Process masks
        for mask_file in glob.glob(os.path.join(city_mask_folder, "*_gtFine_*.*")):
            base_name = os.path.basename(mask_file)
            
            # Keep only _gtFine_labelIds.png files
            if not base_name.endswith("_gtFine_labelIds.png"):
                os.remove(mask_file)
                continue
            
            # Check if the corresponding image exists
            mask_base_name = base_name.replace("_gtFine_labelIds.png", "")
            if mask_base_name not in valid_images:
                os.remove(mask_file)

def visualize_batch(images, targets, preds, ignore_idx=[255], ignore_pred=False):
    """
    Visualizes a batch of images, their ground truth labels, and predictions,
    while ignoring specified pixels in the target mask.

    Args:
        images (Tensor): Batch of images (B, C, H, W), uint8.
        targets (Tensor): Batch of ground truth masks (B, H, W), int.
        preds (Tensor): Batch of predicted masks (B, H, W), int.
        ignore_index (int, optional): Pixel value to ignore in target visualization. Default is 255.
    """
    batch_size = images.shape[0]
    
    fig, axes = plt.subplots(batch_size, 3, figsize=(10, 5 * batch_size))
    
    if batch_size == 1:
        axes = [axes]  # Ensure axes is iterable when batch_size is 1
    
    for i in range(batch_size):
        img = images[i].permute(1, 2, 0).cpu().numpy()
        target = targets[i].cpu().numpy()
        pred = preds[i].cpu().numpy()
        
        # Mask out ignored pixels in target
        target_visual = target.copy()
        for idx in ignore_idx:
            target_visual[target_visual == idx] = 0
            if ignore_pred:
                pred[pred == idx] = 0
                
        axes[i][0].imshow(img)
        axes[i][0].set_title("Input Image")
        axes[i][0].axis("off")
        
        axes[i][1].imshow(target_visual)  # Adjust colormap if needed
        axes[i][1].set_title("Ground Truth")
        axes[i][1].axis("off")
        
        axes[i][2].imshow(pred)  # Adjust colormap if needed
        axes[i][2].set_title("Prediction")
        axes[i][2].axis("off")
    
    plt.tight_layout()
    plt.show()