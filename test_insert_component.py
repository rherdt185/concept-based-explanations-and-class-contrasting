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
from example_generate_basis_visualization_from_data_imagenet import sample_closest_image_patches, generate_data




model_name="resnet50_robust"
layer_name="_layer3[5]"

model = load_model(model_name).eval().to(DEVICE)
layer = model.layer3[5]

def get_concept_patches(model, layer_name, activations, image_dataset, component_index=0):
    folder_path = os.path.join(OUTPUT_PATH, "contrasting", model_name + layer_name)
    class_combination_folder = "249_250_no_ood"
    basis_path = os.path.join(folder_path, class_combination_folder, "nmf", "nmf_basis.pt")
    activation_vectors = torch.load(basis_path).to(DEVICE)



    concept_patches = sample_closest_image_patches(activation_vectors[component_index], activations, image_dataset, n_to_sample=1)

    return concept_patches


dataset = get_dataset(use_train_ds=False)
dataloader = DataLoader(dataset, shuffle=False, batch_size=128, num_workers=8)

activations = generate_data(model, dataloader, kernel_size=7).to(DEVICE)
image_dataset = get_dataset(return_original_sample=True, use_train_ds=False)



other_class = 250

dataset = get_dataset_for_class(model_name=model_name, class_idx=other_class)
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

print("pred malamute: {}".format(torch.mean(pred_malamute)))
print("pred husky: {}".format(torch.mean(pred_husky)))




for i in range(6):
    concept_patches = get_concept_patches(model, layer_name, activations, image_dataset, component_index=i)
    concept_patch = torch.cat([concept_patch for concept_patch in concept_patches], dim=1).unsqueeze(dim=0)

    print(concept_patch.shape)

    for imgs in dataloader:
        imgs = imgs[0].to(DEVICE)
        #imgs[:, :, -concept_patch.shape[2]:, -concept_patch.shape[3]:] = concept_patch
        imgs[:, :, -56:, -56:] = concept_patch

        #imgs[:, :, -160:, -60:] = person_crop
        #print(imgs.shape)
        with torch.no_grad():
            pred = F.softmax(model(imgs), dim=1)
            pred_malamute = pred[:, 249]
            pred_husky = pred[:, 250]

    print("pred malamute: {}".format(torch.mean(pred_malamute)))
    print("pred husky: {}".format(torch.mean(pred_husky)))


