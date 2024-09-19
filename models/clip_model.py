import torch
import clip
from open_clip import create_model_from_pretrained
import yaml
import streamlit as st

# Load configuration
with open("config/config.yaml", 'r') as f:
    config = yaml.safe_load(f)

class ModelLoader:
    @staticmethod
    @st.cache_resource
    def load_clip_model(model_name, device):
        if model_name == "ViT-B/32":
            model, preprocess = clip.load(model_name, device=device)
        else:
            model_path = config["model_files"][model_name]
            model, preprocess = create_model_from_pretrained(f"hf-hub:{model_path}")
            model = model.to(device)
        return model, preprocess