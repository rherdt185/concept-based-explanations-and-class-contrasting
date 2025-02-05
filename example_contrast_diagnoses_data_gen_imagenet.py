from torch.utils.data import DataLoader, TensorDataset
import os
from tqdm import tqdm
import numpy as np
from pathlib import Path

from core.core import extract_attribution_filtered_activation_vectors, collect_patches
from core.imagenet_utils import get_dataset_for_class, load_model, generate_prediction_list, get_dataset
from core.config import DEVICE, OUTPUT_PATH

from core.separate_features import train_linear_classifiers





def generate_data(model_name="resnet50_robust", layer_name="layer3[5]"):
    for class_idx in tqdm(range(1000), ascii=True):
        dataset = get_dataset_for_class(model_name=model_name, class_idx=class_idx)
        if dataset is not None:
            dataloader = DataLoader(dataset, shuffle=True, batch_size=1, num_workers=1)

            model = load_model(model_name).eval().to(DEVICE)
            #print(model)
            layer = eval("model."+layer_name) #model.layer3[2]

            patches = collect_patches(dataloader)
            if len(patches) > 60:
                patches = patches[:60]
            if len(patches) > 0:
                dataset = TensorDataset(patches)

                save_folder_path = os.path.join(OUTPUT_PATH, "activations_for_gmm_training", model_name + "_" + layer_name)

                above_cutoff_activations = extract_attribution_filtered_activation_vectors(dataset, target_channel=class_idx,
                                                model=model, wrapped_model=model, layer=layer,
                                                batch_size_attribution=16, batch_size_activation=256,
                                                attribution_cutoff=0.25)

                save_name = os.path.join(save_folder_path, str(class_idx) + ".npy")
                Path(save_folder_path).mkdir(parents=True, exist_ok=True)
                np.save(save_name, above_cutoff_activations.cpu().numpy())


def run_data_generation(model_name="resnet50_robust", layer_name="layer3[5]"):
    #dataset = get_dataset(return_original_sample=False, use_train_ds=False)
    #generate_prediction_list(model_name=model_name, dataset=dataset)

    generate_data(model_name=model_name, layer_name=layer_name)

    train_linear_classifiers(activation_vecs_folder=os.path.join(OUTPUT_PATH, "activations_for_gmm_training", model_name + "_" + layer_name),
                            output_folder=os.path.join(OUTPUT_PATH, "hyperplanes_from_attribution", model_name + "_" + layer_name),
                            features_list=[i for i in range(1000)])

#run_data_generation(model_name="resnet50_robust", layer_name="layer3[5]")
#run_data_generation(model_name="resnet50_robust", layer_name="layer4[2]")
#run_data_generation(model_name="resnet50_robust", layer_name="layer2[3]")

run_data_generation(model_name="resnet50", layer_name="layer3[5]")
run_data_generation(model_name="resnet34", layer_name="layer3[5]")
run_data_generation(model_name="resnet50_robust", layer_name="layer3[5]")


#generate_data("resnet50")
#generate_data("resnet34")


#train_linear_classifiers(activation_vecs_folder=os.path.join(OUTPUT_PATH, "activations_for_gmm_training", "resnet50"),
#                         output_folder=os.path.join(OUTPUT_PATH, "hyperplanes_from_attribution", "resnet50"),
#                         features_list=[i for i in range(1000)])


