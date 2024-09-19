import torch
import os
import streamlit as st
import yaml
import h5py
import numpy as np
import pandas as pd

# Load configuration
with open("config/config.yaml", 'r') as f:
    config = yaml.safe_load(f)

class DataLoader:
    @staticmethod
    @st.cache_data
    def load_data(model_name, file_type="pt", device="cuda"):
        model_file = config['model_files'][model_name].split("/")[-1]
        
        if file_type == "pt":
            features_file = os.path.join(config['data_dir'], f"image_features_{model_file}.pt")
            ids_file = os.path.join(config['data_dir'], f"item_ids_{model_file}.pt")
            
            image_features = torch.load(features_file, map_location=device)
            item_ids = torch.load(ids_file, map_location=device)
        
        elif file_type == "h5":
            features_file = os.path.join(config['data_dir'], f"image_features_{model_file}.h5")
            
            # Open the HDF5 file in read mode
            with h5py.File(features_file, 'r') as h5f:
                # Load embeddings
                image_features = h5f['features'][:]
                image_features = torch.tensor(image_features.astype(np.float32))
                image_features = image_features.to(device)
                
                # Load IDs
                item_ids = h5f['ids'][:]
                item_ids = [id.decode('utf-8') if isinstance(id, bytes) else id for id in item_ids]
                
        elif file_type == "feather":
            df = pd.read_feather("/mnt/md0/projects/image_search/tags_vector/000.feather")
            item_ids = df["item_id"].values
            image_features = df["vector_clip_vit_b_32"]
        
        else:
            raise ValueError("Unsupported file type. Please choose 'pt' or 'h5'.")
        
        return image_features, item_ids