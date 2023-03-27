import os
import clip
import torch
from torchvision.datasets import CIFAR100
from PIL import Image

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

image_input = preprocess(Image.open("HB1.jpg")).unsqueeze(0).to(device)
list = ('red', 'envelope', 'China', 'Red Envelope')
text = clip.tokenize([(f"a photo of a {c}.") for c in list]).to(device)

# Calculate features
with torch.no_grad():
    image_features = model.encode_image(image_input)
    text_features = model.encode_text(text)

# Pick the top 5 most similar labels for the image
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
values, indices = similarity[0].topk(4)

# Print the result
print("\nTop predictions:\n")
for value, index in zip(values, indices):
    print(f"{list[index]:>16s}: {100 * value.item():.2f}%")