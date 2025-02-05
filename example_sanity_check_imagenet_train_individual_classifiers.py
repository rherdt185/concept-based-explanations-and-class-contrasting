
from torch.utils.data import DataLoader, TensorDataset
import os
from tqdm import tqdm
import numpy as np
import torch
from pathlib import Path
import random

from core.core import collect_patches
from core.sanity_checks import test_move_along_hyperplane
from core.imagenet_utils import get_dataset_for_class, load_model
from core.config import DEVICE, OUTPUT_PATH
from core.linear_classifier import train_linear_classifier

offsets = [0.0, -0.5, -1.0, -2.0, -10.0, -15.0, -20.0, -35.0, -50.0, -200.0]
#offsets = [0.0, -200.0, -500.0, -2000.0, -5000.0]

all_adjusted_preds = []#torch.zeros((1000, 1000, len(offsets)))
tested_combinations = set()

model_name = "resnet50_robust"
model_name_loading_vecs = "resnet50_robust_layer3[5]"
model = load_model(model_name).eval().to(DEVICE)
layer = model.layer3[5]

"""
model_name = "vit_b_32"
model_name_loading_vecs = "vit_b_32_encoder.layers.encoder_layer_7"
model = load_model(model_name).eval().to(DEVICE)
layer = model.encoder.layers.encoder_layer_7
"""

for i in tqdm(range(1000)):
    #for class_idx in tqdm(range(1000), ascii=True):
    #if class_idx != 1:
    #    continue
    #if class_idx == 0:
    #    continue

    if i != 249:
        continue

    class_idx = 250#random.randint(0, 150)

    dataset = get_dataset_for_class(model_name=model_name, class_idx=class_idx)
    if dataset is not None:
        dataloader = DataLoader(dataset, shuffle=True, batch_size=1, num_workers=1)

        patches = collect_patches(dataloader)
        if len(patches) > 60:
            patches = patches[:60]
        if len(patches) > 0:
            dataset = TensorDataset(patches)
            dataloader = DataLoader(dataset, shuffle=False, batch_size=1, num_workers=1)

            activations = torch.from_numpy(np.load(os.path.join(OUTPUT_PATH, "activations_for_gmm_training", model_name_loading_vecs, str(class_idx) + ".npy"))).to("cuda")


        for j in range(1):
            #for other_class in tqdm(range(1000), ascii=True):
            other_class = 249#random.randint(1, 150)
            #continue
            #if other_class == 0:
            #    continue
            if class_idx == other_class:
                continue
            if (class_idx, other_class) in tested_combinations:
                continue
            tested_combinations.add((class_idx, other_class))

            activations_other = torch.from_numpy(np.load(os.path.join(OUTPUT_PATH, "activations_for_gmm_training", model_name_loading_vecs, str(other_class) + ".npy"))).to("cuda")

            training_activation_vecs = torch.cat([activations, activations_other])

            targets = torch.zeros(len(activations), device="cuda")
            targets_other_feature = torch.ones(len(activations_other), device="cuda")
            training_targets = torch.cat([targets, targets_other_feature], dim=0)

            hyperplane_normal, hyperplane_bias, accuracy = train_linear_classifier(training_activation_vecs, training_targets)

            adjusted_preds, max_adjusted_preds = test_move_along_hyperplane(dataloader, model, layer, hyperplane_normal, from_class=class_idx, other_class=other_class,
                                                                            do_prints=False)

            print("other preds: {}".format(adjusted_preds[:, other_class]))
            print("original preds: {}".format(adjusted_preds[:, class_idx]))

            all_adjusted_preds.append(torch.tensor(adjusted_preds))
            #all_adjusted_preds_dict[str(class_idx) + "_" + str(other_class)] = adjusted_preds
            #all_adjusted_preds_dict["max_" + str(class_idx) + "_" + str(other_class)] = max_adjusted_preds

        #break

#Path(os.path.join(OUTPUT_PATH, "sanity_check", "resnet50_robust")).mkdir(parents=True, exist_ok=True)
#all_adjusted_preds = torch.stack(all_adjusted_preds, dim=0)
#out_file = os.path.join(OUTPUT_PATH, "sanity_check", "resnet50_robust", "adjusted_preds.npy")
#np.save(out_file, all_adjusted_preds.cpu().numpy())
