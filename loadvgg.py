import torchvision.models as models
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

models.vgg19(pretrained=True).features.to(device).eval()

print("vgg loaded")

