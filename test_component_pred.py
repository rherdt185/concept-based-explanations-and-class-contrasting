from torch.utils.data import DataLoader, TensorDataset
import os
from tqdm import tqdm
import numpy as np
from pathlib import Path
import torch
import torch.nn.functional as F
import torchvision

from core.core import extract_attribution_filtered_activation_vectors, collect_patches
from core.imagenet_utils import get_dataset_for_class, load_model, generate_prediction_list, get_dataset, get_dataset_excluding_class
from core.config import DEVICE, OUTPUT_PATH

from core.separate_features import train_linear_classifiers
from example_generate_basis_visualization_from_data_imagenet import sample_closest_image_patches, generate_data, generate_data_exclude_class




model_name="resnet50_robust"
layer_name="_layer3[5]"

model = load_model(model_name).eval().to(DEVICE)
layer = model.layer3[5]

def get_concept_patches(activation_vectors, activations, image_dataset):
    all_concept_patches = []
    for component_index in range(len(activation_vectors)):
        concept_patches = sample_closest_image_patches(activation_vectors[component_index], activations, image_dataset, n_to_sample=1, is_image_dataset_list=True)
        all_concept_patches.append(concept_patches)

    return torch.cat(all_concept_patches, dim=2)


#dataset = get_dataset_excluding_class(model_name, class_idx=3)
dataset = get_dataset(use_train_ds=False)
dataloader = DataLoader(dataset, shuffle=False, batch_size=128, num_workers=8)

#activations, images = generate_data_exclude_class(model, layer, dataloader, kernel_size=2, class_to_exclude=249)


#print(activations.shape)
#print(images.shape)

activations = generate_data(model, layer, dataloader, kernel_size=2)#.to(DEVICE)
image_dataset = dataset
#image_dataset = get_dataset(return_original_sample=True, use_train_ds=False)



#folder_path = os.path.join(OUTPUT_PATH, "contrasting", model_name + layer_name)
#class_combination_folder = "249_250_no_ood"

folder_path = os.path.join(OUTPUT_PATH, model_name + layer_name)
class_combination_folder = "249"
basis_path = os.path.join(folder_path, class_combination_folder, "nmf", "nmf_basis.pt")
activation_vectors = torch.load(basis_path).to(DEVICE)

concept_patches = get_concept_patches(activation_vectors, activations, image_dataset).to(DEVICE) #
concept_patches = torch.cat([concept_patches[i] for i in range(len(concept_patches))], dim=-1).unsqueeze(dim=0)
#concept_patches = torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(concept_patches).to(DEVICE)

print("concept patches shape: {}".format(concept_patches.shape))

with torch.no_grad():
    pred = F.softmax(model(concept_patches), dim=1)
    pred_malamute = pred[:, 249]
    pred_husky = pred[:, 250]

    print("pred malamute: {}".format(torch.mean(pred_malamute)))
    print("pred husky: {}".format(torch.mean(pred_husky)))

    #print(torch.argmax(pred))
    print(torch.argmax(torch.mean(pred, dim=0)))
    print(torch.max(torch.mean(pred, dim=0)))


