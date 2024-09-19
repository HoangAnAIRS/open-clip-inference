import torch
import numpy as np
from collections import OrderedDict
from torch.nn.functional import cosine_similarity
import clip

def process_query(query, model, device="cuda"):
    with torch.no_grad():
        text_features = model.encode_text(clip.tokenize([query]).to(device))
        print(text_features.dtype)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features

def compute_distances(image_features, text_vector, metric='cosine'):
    if metric == 'cosine':
        return cosine_similarity(image_features, text_vector).squeeze().cpu().numpy()
    elif metric == 'l2':
        return torch.norm(image_features - text_vector, dim=1).squeeze().cpu().numpy()
    else:
        raise ValueError(f"Unknown metric: {metric}")

def process_data(image_features, item_ids, text_vector, metric='cosine', limit=200):
    distance = compute_distances(image_features, text_vector, metric)
    top_indices = np.argsort(distance)[::-1 if metric == 'cosine' else 1][:limit]
    top_item_ids = [item_ids[i] for i in top_indices]
    top_distance = distance[top_indices].tolist()
    
    return OrderedDict(zip(map(str, top_item_ids), top_distance))
