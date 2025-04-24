import json 
from PIL import Image 
from transformers import CLIPProcessor

class ClipEmbedder():
    def __init__(self, annotation_file: str, processor: CLIPProcessor): 
        with open(annotation_file, 'r') as f: 
            self.annotations = json.load(f) 

        self.image_filenames = list(self.annotations.keys()) 
        self.processor = processor 

    def __len__(self): 
        return len(self.image_filenames)
    
    def __getitem__(self, idx): 
        img_path = self.image_filenames[idx] 
        text = self.annotations[img_path]

        image = Image.open(img_path).convert("RGB")

        inputs = self.processor(
            text=[text],
            images=image,
            return_tensors="pt",
            padding=True
        )

        return {
            'pixel_values': inputs['pixel_values'].squeeze(0),
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0)
        }