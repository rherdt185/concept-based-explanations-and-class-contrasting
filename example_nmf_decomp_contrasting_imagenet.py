from torch.utils.data import DataLoader, TensorDataset

from tqdm import tqdm
import os
import torch
import pickle

from core.core import nmf_decomp_hyperplane_ood_filtered_activations, collect_patches
from core.imagenet_utils import get_dataset_for_class, load_model
from core.config import DEVICE, OUTPUT_PATH




def do_nmf_decomp(model_name, layer_name, class_idx, other_class):
    hyperplane_folder = os.path.join(OUTPUT_PATH, "hyperplanes_from_attribution", model_name + layer_name)

    dataset = get_dataset_for_class(model_name=model_name, class_idx=class_idx)

    hyperplane_normals = torch.load(os.path.join(hyperplane_folder, str(class_idx) + ".pt"))
    hyperplane_biases = torch.load(os.path.join(hyperplane_folder, "bias_" + str(class_idx) + ".pt"))

    hyperplane_normal = -hyperplane_normals[other_class].to(DEVICE)
    hyperplane_bias = hyperplane_biases[other_class].to(DEVICE)

    if dataset is not None:
        if len(dataset) > 0:
            dataloader = DataLoader(dataset, shuffle=True, batch_size=1, num_workers=1)

            model = load_model(model_name).eval().to(DEVICE)
            layer = model.layer3[5]

            patches = collect_patches(dataloader)
            if len(patches) > 60:
                patches = patches[:60]
            if len(patches) > 0:
                dataset = TensorDataset(patches)
                # batch size has to be 1, code does not produce correct results with larger batch sizes yet (assumes batch size to be 1)
                dataloader = DataLoader(dataset, shuffle=False, batch_size=1, num_workers=1)

                save_folder_path = os.path.join(OUTPUT_PATH, "contrasting", model_name + layer_name, str(class_idx) + "_" + str(other_class) + "_no_ood")

                """
                gmm_path = os.path.join(OUTPUT_PATH, "gmm_models", model_name, "10__" + str(class_idx) + ".pkl")
                #gmm_path = os.path.join("/home/digipath2/projects/xai/digipath_xai_fast_api/gmm_models_diff_comps", "10__" + str(class_idx) + "_.pkl")
                with open(gmm_path, "rb") as file:
                    gmm_model = pickle.load(file)
                """

                nmf_decomp_hyperplane_ood_filtered_activations(dataloader, model=model, layer=layer,
                                                            hyperplane_normal=hyperplane_normal,
                                                            hyperplane_bias=hyperplane_bias,
                                                            hyperplane_additional_bias=0.0,
                                                save_folder=save_folder_path, n_nmf_components=6,
                                                gmms=None)#[gmm_model])





if __name__ == "__main__":
    model_name = "resnet50_robust"
    layer_name = "_layer3[5]"

    class_combinations = [[249, 250], [250, 249]]
    #class_combinations = [[248, 249], [249, 248], [248, 250], [250, 248]]

    for class_idx, other_class in class_combinations:
        #class_idx = 1
        #other_class = 32
        do_nmf_decomp(model_name, layer_name, class_idx)


