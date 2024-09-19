import torch
import torch.nn.functional as F
from PIL import Image
import clip
import os
from tqdm import tqdm
import time
from loguru import logger
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from open_clip import create_model_from_pretrained, get_tokenizer
import yaml

# Load configuration from YAML file
with open("config/encoder.yaml", 'r') as f:
    config = yaml.safe_load(f)
    
root = config["root"]
images = [os.path.join(root, file) for file in os.listdir(root)]
DEVICE = config["device"]
output = config["output"]
model_name = config["model_name"]

# Custom Dataset class with caching
class ImageDataset(Dataset):
    def __init__(self, image_paths, preprocess, cache_size=1000):
        self.image_paths = image_paths
        self.preprocess = preprocess
        self.cache = {}
        self.cache_size = cache_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image_id = image_path.split("/")[-1].split(".")[0]
        
        if image_path in self.cache:
            return image_id, self.cache[image_path]
        
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.preprocess(image)
        
        if len(self.cache) < self.cache_size:
            self.cache[image_path] = image_tensor
        
        return image_id, image_tensor

def main():    
    t1 = time.time()
    
    logger.info(f"Loading model: {model_name}")
    if model_name == "ViT-B/32":
        model, preprocess = clip.load(model_name, device=DEVICE)
    else:
        model, preprocess = create_model_from_pretrained(f"hf-hub:{model_name}")  
    t2 = time.time()
    logger.info(f"Time to load model: {t2-t1}")
    logger.info(f"Model: {model_name}")

    model.eval()  # Set the model to evaluation mode
    
    model = model.to(DEVICE)

    # Enable cuDNN autotuner
    torch.backends.cudnn.benchmark = True

    # Set batch size
    batch_size = 8  # Increased batch size

    # Create DataLoader
    dataset = ImageDataset(images, preprocess)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=8, pin_memory=True, persistent_workers=True)

    # Preallocate memory for results
    num_images = len(images)
    feature_dim = model.visual.output_dim
    image_features_tensor = torch.zeros((num_images, feature_dim), dtype=torch.float32)
    item_ids = np.empty(num_images, dtype=object)

    # Process images in batches
    start_idx = 0
    for batch_ids, batch_inputs in tqdm(dataloader):
        batch_inputs = batch_inputs.to(DEVICE, non_blocking=True)
        
        with torch.no_grad():  # Enable automatic mixed precision
            batch_features = model.encode_image(batch_inputs)
        
        batch_features /= batch_features.norm(dim=-1, keepdim=True)
        
        end_idx = start_idx + len(batch_ids)
        image_features_tensor[start_idx:end_idx] = batch_features.cpu()
        item_ids[start_idx:end_idx] = batch_ids
        start_idx = end_idx

    # Save the results
    backbone  = "openai-ViT-B-32" if model_name == "ViT-B/32" else model_name.split("/")[-1]
    
    torch.save(image_features_tensor, f"{output}/image_features_{backbone}.pt")
    print("Successfully saved image features with name: ", f"images_features_{backbone}.pt")
    torch.save(item_ids, f"{output}/item_ids_{backbone}.pt")
    print("Successfully saved item ids with name: ", f"item_ids_{backbone}.pt")

if __name__ == "__main__":
    main()