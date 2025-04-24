import clip
import torch
import torch.nn.functional as F

from PIL import Image
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from .clip_dataset import CLIPDataset
from config import CLIP_MODEL

def train(model, dataloader, optimizer, device, epochs=3):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for images, texts in tqdm(dataloader, desc=f"Epoch {epoch + 1}"):
            images = images.to(device)
            texts = clip.tokenize(texts).to(device)

            logits_per_image, logits_per_text = model(images, texts)
            ground_truth = torch.arange(len(images), dtype=torch.long, device=device)

            # Contrastive loss 
            loss = (F.cross_entropy(logits_per_image, ground_truth) +
                    F.cross_entropy(logits_per_text, ground_truth)) / 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}: Loss = {avg_loss:.4f}")


def main():
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(CLIP_MODEL, device=device)

    # Create dataset w/ processed images + geocaptions
    S3 = True 

    if S3: 
        annotation_file = "s3://animaldex/data/inaturalist_2017/processed/geocaptions.json"
    else: 
        annotation_file = "data/inaturalist_2017/processed/geocaptions.json"

    dataset = CLIPDataset(annotation_file=annotation_file, preprocess=preprocess)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=5e-6)

    train(model, dataloader, optimizer, device)

    # Store model weights
    torch.save(model.state_dict(), "src/models/finetuned_clip_subset.pt")

if __name__ == "__main__":
    main()
