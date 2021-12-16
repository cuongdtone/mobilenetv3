from torchvision import models
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import cv2


class mobilenetv3:
    def __init__(self, weights, use_cuda=False):
        self.device = torch.device('cuda:0' if use_cuda else 'cpu')
        self.model = models.mobilenet_v3_small(pretrained=False).to(self.device)
        self.model.classifier = nn.Sequential(
            nn.Linear(576, 128),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(128, 2)).to(self.device)
        self.weights = weights
        self.load_model()
    def load_model(self):
        self.model.load_state_dict(torch.load(self.weights, map_location=self.device))
    def preprocess_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(image)
        # img = Image.open(i).convert('RGB')
        input_size = 112
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        preprocess = transforms.Compose([
            transforms.Resize(int(input_size / 0.875)),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            normalize])
        img_preprocessed = preprocess(img)
        batch_img_tensor = torch.unsqueeze(img_preprocessed, 0)
        return batch_img_tensor
    def predict(self, image):
        img = self.preprocess_image(image).to(self.device)
        self.model.eval()
        out = self.model(img)
        _, index = torch.max(out, 1)
        #percentage = (nn.functional.softmax(out, dim=1)[0]).tolist()
        return index.numpy()[0]

