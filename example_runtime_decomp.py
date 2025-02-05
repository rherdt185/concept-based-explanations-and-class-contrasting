
import torch
import torch.nn.functional as F
import torchvision
import os

from pathlib import Path

from torch.utils.data import DataLoader, TensorDataset

from tqdm import tqdm

from core.utils import access_activations_forward_hook
from core.core import nmf_attribution_whole_ds_decomp, extract_overlay_channel_activations, generate_mask_from_gmm, compute_attribution
from core.digipath_utils import collect_dataset, get_model, collect_patches_from_prediction, AttributionLimitToTargetClassWrapper
from core.config import DEVICE, OUTPUT_PATH

import pickle

from core.runtime import decompose_from_nmf_basis

class_idx = 1
n_components = 6

folder_path = os.path.join(OUTPUT_PATH, "version_299_layer3_nmf_decomp")
nmf_path = os.path.join(folder_path, str(class_idx), "vectors", "nmf_basis.pt")

output_dir = os.path.join(OUTPUT_PATH, "tmp", "runtime_nmf_decomp", "attribution_filtered")

for i in range(n_components):
    Path(os.path.join(output_dir, str(class_idx), "patches", str(i))).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(output_dir, str(class_idx), "patches_original", str(i))).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(output_dir, str(class_idx), "masks", str(i))).mkdir(parents=True, exist_ok=True)

nmf_basis = torch.load(nmf_path).float()*1000.0

gmm_path = os.path.join(OUTPUT_PATH, "gmm_models", "version_299", "10__" + str(class_idx) + ".pkl")
#gmm_path = os.path.join("/home/digipath2/projects/xai/digipath_xai_fast_api/gmm_models_diff_comps", "10__" + str(class_idx) + "_.pkl")
with open(gmm_path, "rb") as file:
    gmm_model = pickle.load(file)


dataset = collect_dataset(target_class=class_idx)
if dataset is not None:
    if len(dataset) > 0:
        dataloader = DataLoader(dataset, shuffle=True, batch_size=1, num_workers=1)

        model = get_model("version_299").eval().to(DEVICE)
        layer = getattr(model.model.encoder.down_blocks, 'down block 3').resnet_blocks[2]

        wrapped_model = AttributionLimitToTargetClassWrapper(model, target_class=class_idx)

        patches = collect_patches_from_prediction(dataloader, model, target_channel=class_idx, num_to_sample=6)

        image_index = 0
        for patch in patches:
            patch = patch.unsqueeze(dim=0).to(DEVICE)

            dataloader_attribution = DataLoader(TensorDataset(patch.cpu()), shuffle=False, batch_size=1)

            activations = access_activations_forward_hook([patch], model, layer)
            attribution = compute_attribution(dataloader_attribution, wrapped_model, class_idx, layer)

            attribution = torch.mean(F.relu(attribution), dim=1).unsqueeze(dim=1)

            activations = activations#*attribution

            #in_distribution_mask = generate_mask_from_gmm(activations, [gmm_model], gmm_cutoff_score=-300.0)


            #save_folder_path = OUTPUT_PATH + "version_299_layer3_nmf_decomp"

            embedded, final_reconstruction = decompose_from_nmf_basis(activations, nmf_basis)

            embedded = embedded*attribution[0]

            #embedded = embedded*in_distribution_mask[0]

            print(torch.mean(embedded[0]))
            print(torch.std(embedded[0]))

            print(embedded.shape)
            channel_activation = extract_overlay_channel_activations(
                embedded.unsqueeze(dim=0), patch, [j for j in range(6)]
            )



            for k, overlay in enumerate(channel_activation):
                mask = embedded[k]
                #print("activation vec shape: {}".format(activation_vec.shape))
                mask = mask - torch.min(mask)
                mask = mask / torch.max(mask)
                #print(mask.shape)
                torchvision.utils.save_image(overlay, os.path.join(output_dir, str(class_idx), "patches", str(k%n_components), str(image_index) + ".jpg"))
                torchvision.utils.save_image(patch, os.path.join(output_dir, str(class_idx), "patches_original", str(k%n_components), str(image_index) + ".jpg"))
                torchvision.utils.save_image(mask, os.path.join(output_dir, str(class_idx), "masks", str(k%n_components), str(image_index) + ".png"))
            image_index += 1



