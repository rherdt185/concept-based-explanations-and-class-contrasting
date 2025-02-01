from torch.utils.data import DataLoader, TensorDataset
import os
from tqdm import tqdm
import numpy as np
from pathlib import Path
import torch
import torch.nn.functional as F
import torchvision

from core.core import extract_attribution_filtered_activation_vectors, collect_patches
from core.imagenet_utils import get_dataset_for_class, load_model, generate_prediction_list, get_dataset
from core.config import DEVICE, OUTPUT_PATH

from core.separate_features import train_linear_classifiers






model_name="resnet50_robust"


model = load_model(model_name).eval().to(DEVICE)
layer = model.layer3[5]

dataset = get_dataset_for_class(model_name=model_name, class_idx=249)
dataloader = DataLoader(dataset, shuffle=False, batch_size=1, num_workers=1)
patches_malamute = collect_patches(dataloader)
#if len(patches_malamute) > 60:
#    patches_malamute = patches_malamute[:60]

img = patches_malamute[24].unsqueeze(dim=0)
person_crop = img[:, :, 20:180, 135:195]

torchvision.utils.save_image(person_crop, "test_img_person.jpg")



dataset = get_dataset_for_class(model_name=model_name, class_idx=250)
dataloader = DataLoader(dataset, shuffle=False, batch_size=1, num_workers=1)
patches = collect_patches(dataloader)
if len(patches) > 60:
    patches = patches[:60]

dataset = TensorDataset(patches)
dataloader = DataLoader(dataset, shuffle=False, batch_size=100, num_workers=8)

for imgs in dataloader:
    imgs = imgs[0].to(DEVICE)
    print(imgs.shape)
    with torch.no_grad():
        pred = F.softmax(model(imgs), dim=1)
        pred_malamute = pred[:, 249]
        pred_husky = pred[:, 250]

print("original predictions")
print("pred malamute: {}".format(torch.mean(pred_malamute)))
print("pred husky: {}".format(torch.mean(pred_husky)))


#person_crop = torch.zeros_like(person_crop)

for imgs in dataloader:
    imgs = imgs[0].to(DEVICE)
    imgs[:, :, -person_crop.shape[2]:, -person_crop.shape[3]:] = person_crop
    #print(imgs.shape)
    with torch.no_grad():
        pred = F.softmax(model(imgs), dim=1)
        pred_malamute = pred[:, 249]
        pred_husky = pred[:, 250]

print("adding a person image")
print("pred malamute: {}".format(torch.mean(pred_malamute)))
print("pred husky: {}".format(torch.mean(pred_husky)))


torchvision.utils.save_image(imgs[0].unsqueeze(dim=0), "husky_images_person_inserted.jpg")


for imgs in dataloader:
    imgs = imgs[0].to(DEVICE)
    imgs[:, :, -person_crop.shape[2]:, -person_crop.shape[3]:] = torch.zeros_like(person_crop)
    #print(imgs.shape)
    with torch.no_grad():
        pred = F.softmax(model(imgs), dim=1)
        pred_malamute = pred[:, 249]
        pred_husky = pred[:, 250]

print("adding zeros after normalization instead instead of the person image")
print("pred malamute: {}".format(torch.mean(pred_malamute)))
print("pred husky: {}".format(torch.mean(pred_husky)))

torchvision.utils.save_image(imgs[0].unsqueeze(dim=0), "husky_images_black_inserted.jpg")





"""
img = patches_malamute[23].unsqueeze(dim=0)
person_crop = img[:, :, 5:70, 100:]

torchvision.utils.save_image(person_crop, "test_img_person.jpg")

#person_crop = torch.zeros_like(person_crop)

for imgs in dataloader:
    imgs = imgs[0].to(DEVICE)
    imgs[:, :, -person_crop.shape[2]:, -person_crop.shape[3]:] = person_crop
    print(imgs.shape)
    with torch.no_grad():
        pred = F.softmax(model(imgs), dim=1)
        pred_malamute = pred[:, 249]
        pred_husky = pred[:, 250]

print("pred malamute: {}".format(torch.mean(pred_malamute)))
print("pred husky: {}".format(torch.mean(pred_husky)))
"""




