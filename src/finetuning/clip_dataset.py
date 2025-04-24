import json 

from PIL import Image 
from smart_open import open as s3_open
from torch.utils.data import Dataset

class CLIPDataset(Dataset):
    def __init__(self, annotation_file, preprocess):
        with s3_open(annotation_file, 'r') as f:
            self.annotations = json.load(f)

        self.image_paths = list(self.annotations.keys())
        self.texts = [self.annotations[k] for k in self.image_paths]
        self.preprocess = preprocess

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        with s3_open(image_path, 'rb') as f:
            image = self.preprocess(Image.open(f).convert("RGB"))
        
        text = self.texts[idx]
        
        return image, text 
    
  