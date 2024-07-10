import torch
from torchvision import models, transforms
from PIL import Image
import json

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = models.__dict__[checkpoint['arch']](pretrained=True)
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model

def process_image(image):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = preprocess(image)
    return image

def predict(image_path, model, topk, category_names):
    model.eval()
    image = Image.open(image_path)
    image = process_image(image)
    image = image.unsqueeze(0)
    if torch.cuda.is_available():
        model.cuda()
        image = image.cuda()
    with torch.no_grad():
        output = model(image)
        probabilities = torch.exp(output)
        top_probabilities, top_indices = torch.topk(probabilities, topk)
        top_probabilities = top_probabilities.squeeze().tolist()
        top_indices = top_indices.squeeze().tolist()
    idx_to_class = {value: key for key, value in model.class_to_idx.items()}
    if category_names:
        with open(category_names, 'r') as f:
            cat_to_name = json.load(f)
        top_classes = [cat_to_name[idx_to_class[idx]] for idx in top_indices]
    else:
        top_classes = [idx_to_class[idx] for idx in top_indices]
    return top_probabilities, top_classes

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Image Classifier - Prediction')
    parser.add_argument('image_path', type=str, help='path to the image file')
    parser.add_argument('checkpoint', type=str, help='path to the checkpoint file')
    parser.add_argument('--top_k', type=int, default=5, help='number of top classes to display')
    parser.add_argument('--category_names', type=str, default=None, help='path to the JSON file containing category names')
    args = parser.parse_args()

    model = load_checkpoint(args.checkpoint)
    top_probabilities, top_classes = predict(args.image_path, model, args.top_k, args.category_names)
    print('Top Probabilities:', top_probabilities)
    print('Top Classes:', top_classes)