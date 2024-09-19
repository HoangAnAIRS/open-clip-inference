import os
import time
import yaml
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from open_clip import create_model_from_pretrained
from tqdm import tqdm
import numpy as np
import h5py

# Load configuration from YAML file
with open("config/encoder.yaml", 'r') as f:
    config = yaml.safe_load(f)

root = config["root"]
output = config["output"]
model_name = config["model_name"]
DEVICE = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")

# Filter image files
valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
images = [os.path.join(root, file) for file in os.listdir(root)
          if file.lower().endswith(valid_extensions)]

# Custom Dataset class
class ImageDataset(Dataset):
    def __init__(self, image_paths, preprocess):
        self.image_paths = image_paths
        self.preprocess = preprocess

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image_id = os.path.splitext(os.path.basename(image_path))[0]

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            image = Image.new('RGB', (224, 224))

        image_tensor = self.preprocess(image)
        return image_id, image_tensor

def main():
    t1 = time.time()

    # Load the model
    print(f"Loading model: {model_name}")
    model, preprocess = create_model_from_pretrained(f"hf-hub:{model_name}")
    model = model.to(DEVICE)
    model.eval()

    t2 = time.time()
    print(f"Time to load model: {t2 - t1:.2f} seconds")

    # Enable cuDNN autotuner
    torch.backends.cudnn.benchmark = True

    # Set batch size
    batch_size = config.get("batch_size", 128)

    # Create DataLoader
    dataset = ImageDataset(images, preprocess)
    num_images = len(dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=8,
                            pin_memory=True, prefetch_factor=2)

    # Get feature dimensions
    feature_dim = model.visual.output_dim

    # Prepare output HDF5 file
    backbone = model_name.split("/")[-1]
    dtype = 'float32'
    hdf5_path = os.path.join(output, f"image_features_{backbone}_{dtype}.h5")

    with h5py.File(hdf5_path, 'w') as h5f:
        features_ds = h5f.create_dataset('features',
                                         shape=(0, feature_dim),
                                         maxshape=(None, feature_dim),
                                         dtype=dtype,
                                         chunks=(batch_size, feature_dim),
                                         compression='gzip')
        ids_dt = h5py.string_dtype(encoding='utf-8')
        ids_ds = h5f.create_dataset('ids',
                                    shape=(0,),
                                    maxshape=(None,),
                                    dtype=ids_dt,
                                    chunks=(batch_size,),
                                    compression='gzip')

        # Process images in batches
        for batch_ids, batch_inputs in tqdm(dataloader):
            batch_inputs = batch_inputs.to(DEVICE, non_blocking=True)

            with torch.no_grad(), torch.cuda.amp.autocast():
                batch_features = model.encode_image(batch_inputs)
                batch_features /= batch_features.norm(dim=-1, keepdim=True)
                batch_features = batch_features.cpu()

            # Append to datasets
            batch_size_actual = batch_features.shape[0]
            features_ds.resize(features_ds.shape[0] + batch_size_actual, axis=0)
            features_ds[-batch_size_actual:] = batch_features
            # get the features_ds and print it

            ids_ds.resize(ids_ds.shape[0] + batch_size_actual, axis=0)
            ids_ds[-batch_size_actual:] = batch_ids

    print(f"Successfully saved features to {hdf5_path}")

if __name__ == "__main__":
    main()