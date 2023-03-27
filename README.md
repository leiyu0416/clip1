# clip1
自然语言编码器得到的向量V是由什么样的自然语言输入得到的？
标签文本，输入一堆标签，形成句子。文本这边就是输入感兴趣的标签有哪些，比如图中的汽车，狗，飞机，鸟等这几个词会变成句子，比如，汽车会变成这是一张汽车的照片，飞机会变成这是一张飞机的照片，几个单词就变成几个句子，然后通过文本编码器会得到这些特征，然后和图像向量去对比会得到相似度。

当标签是'red', 'envelope', 'China'三类时，envelope这个类别概率最大，当继续添加‘Red Envelope’时，预测概率为99.61%，有出现一边倒相信这是Red Envelope的。

# 配置环境依赖的步骤
     1、创建一个新的环境“conda activate clip”
     2、语句命令“conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=10.2”
     3、语句命令“pip install ftfy regex tqdm”
     4、语句命令“pip install git+https://github.com/openai/CLIP.git”
   
# 代码

      import os
      import clip
      import torch
      from torchvision.datasets import CIFAR100
      from PIL import Image
      
     # Load the model
     device = "cuda" if torch.cuda.is_available() else "cpu"
     model, preprocess = clip.load('ViT-B/32', device)
     image_input = preprocess(Image.open("HB1.jpg")).unsqueeze(0).to(device)
     list = ('red', 'envelope', 'China','Red Envelope')
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

The output will look like the following (the exact numbers may be slightly different depending on the compute device):

    Top predictions:

    Red Envelope: 99.61%
        envelope: 0.37%
             red: 0.00%
           China: 0.00%

# 实验结果
当标签是'red', 'envelope', 'China'三类时，envelope这个类别概率最大。
        envelope: 99.37%
             red: 0.41%
           China: 0.22%

当继续添加‘Red Envelope’时，预测概率为99.61%，有出现一边倒相信这是Red Envelope的。
     Red Envelope: 99.61%
        envelope: 0.37%
             red: 0.00%
           China: 0.00%
