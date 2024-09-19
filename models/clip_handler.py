import torch
from models.clip_model import ModelLoader
from utils.data_loader import DataLoader


class CLIPHandler:
    def __init__(self, model_name="ViT-B/32", file_type="pt", device="cuda"):
        self.model, self.preprocess = ModelLoader.load_clip_model(model_name, device)        
        self.image_features, self.item_ids = DataLoader.load_data(model_name, file_type, device)
