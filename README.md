# clip1
自然语言编码器得到的向量V是由什么样的自然语言输入得到的？
标签文本，输入一堆标签，形成句子。文本这边就是输入感兴趣的标签有哪些，比如图中的汽车，狗，飞机，鸟等这几个词会变成句子，比如，汽车会变成这是一张汽车的照片，飞机会变成这是一张飞机的照片，几个单词就变成几个句子，然后通过文本编码器会得到这些特征，然后和图像向量去对比会得到相似度。


import os
import clip
import torch
from torchvision.datasets import CIFAR100
from PIL import Image

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

image_input = preprocess(Image.open("HB1.jpg")).unsqueeze(0).to(device)
list = ('red', 'envelope', 'China')
text = clip.tokenize([(f"a photo of a {c}.") for c in list]).to(device)

# Calculate features
with torch.no_grad():
    image_features = model.encode_image(image_input)
    text_features = model.encode_text(text)

# Pick the top 5 most similar labels for the image
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
values, indices = similarity[0].topk(3)

# Print the result
print("\nTop predictions:\n")
for value, index in zip(values, indices):
    print(f"{list[index]:>16s}: {100 * value.item():.2f}%")


得到的预测值
    Top predictions:

        envelope: 99.37%
             red: 0.41%
           China: 0.22%

