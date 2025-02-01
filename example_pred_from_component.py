from torch.utils.data import DataLoader, TensorDataset
import os
from tqdm import tqdm
import numpy as np
from pathlib import Path
import torch
import torch.nn.functional as F
import torchvision
import random

from core.core import extract_attribution_filtered_activation_vectors, collect_patches
from core.imagenet_utils import get_dataset_for_class, load_model, generate_prediction_list, get_dataset, get_dataset_excluding_class
from core.config import DEVICE, OUTPUT_PATH

from core.separate_features import train_linear_classifiers
from example_generate_basis_visualization_from_data_imagenet import sample_closest_image_patches, generate_data, generate_data_including_prediction

from example_nmf_decomp_contrasting_imagenet import do_nmf_decomp




def get_concept_patches(activation_vectors, activations, image_dataset):
    all_concept_patches = []
    for component_index in range(len(activation_vectors)):
        concept_patches = sample_closest_image_patches(activation_vectors[component_index], activations, image_dataset, n_to_sample=1, is_image_dataset_list=True)
        all_concept_patches.append(concept_patches)

    return torch.cat(all_concept_patches, dim=2)



def get_pred(class_idx, activations, image_dataset, activation_vectors):
    concept_patches = get_concept_patches(activation_vectors, activations, image_dataset).to(DEVICE) #
    concept_patches = torch.cat([concept_patches[i] for i in range(len(concept_patches))], dim=-1).unsqueeze(dim=0)
    #concept_patches = torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(concept_patches).to(DEVICE)

    #print("concept patches shape: {}".format(concept_patches.shape))

    with torch.no_grad():
        pred = F.softmax(model(concept_patches), dim=1)
        pred_class = torch.mean(pred[:, class_idx])
        max_class = torch.argmax(torch.mean(pred, dim=0))
        max_pred = torch.max(torch.mean(pred, dim=0))

        """
        class_preds.append(pred_class)
        if max_class == class_idx:
            matches_desired_pred.append(torch.tensor(1.0))
        else:
            matches_desired_pred.append(torch.tensor(0.0))
        #pred_husky = pred[:, 250]

        #print("pred malamute: {}".format(torch.mean(pred_malamute)))
        #print("pred husky: {}".format(torch.mean(pred_husky)))
        """
        print("class idx: {}".format(class_idx))
        print("pred class: {}, max_class: {}, max_pred: {}".format(pred_class, max_class, max_pred))

        #print(torch.argmax(pred))
        #print(torch.argmax(torch.mean(pred, dim=0)))
        #print(torch.max(torch.mean(pred, dim=0)))



    return pred_class, max_class, max_pred


def run_test(exclude_target_class_from_patches=False, save_output=True, use_average_pooling=True):
    #dataset = get_dataset_excluding_class(model_name, class_idx=3)
    dataset = get_dataset(use_train_ds=False)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=128, num_workers=8)

    activations_all, prediction_classes = generate_data_including_prediction(model, layer, dataloader, kernel_size=2, stride=2, use_average_pooling=use_average_pooling)
    prediction_classes = prediction_classes.cpu().numpy()
    #activations = generate_data(model, dataloader, kernel_size=2).to(DEVICE)

    #image_dataset = get_dataset(return_original_sample=True, use_train_ds=False)

    folder_path = os.path.join(OUTPUT_PATH, model_name + layer_name)
    folder_path_contrasting = os.path.join(OUTPUT_PATH, "contrasting", model_name + layer_name)

    class_preds = []
    matches_desired_pred = []

    contrasting_class_preds = []
    contrasting_matches_desired_pred = []


    for class_idx in tqdm(range(1000), ascii=True):
        #class_idx = random.randint(0, 999)
        #class_idx = 249

        if exclude_target_class_from_patches:
            indices = np.where(prediction_classes != class_idx)[0]
            image_dataset = torch.utils.data.Subset(dataset, indices)
            activations = activations_all[indices]
        else:
            image_dataset = dataset
            activations = activations_all

        #image_dataset = dataset

        class_combination_folder = str(class_idx)
        basis_path = os.path.join(folder_path, class_combination_folder, "nmf", "nmf_basis.pt")
        activation_vectors = torch.load(basis_path).to(DEVICE)

        #activations = activations.to(DEVICE)
        pred_class, max_class, max_pred = get_pred(class_idx, activations, image_dataset, activation_vectors)

        class_preds.append(pred_class)
        if max_class == class_idx:
            matches_desired_pred.append(torch.tensor(1.0))
        else:
            matches_desired_pred.append(torch.tensor(0.0))

            """
            #max_class = 250
            # try with contrasting against maximum predicted class:

            print("contrasting...")

            class_combination_folder = str(int(class_idx)) + "_" + str(int(max_class)) + "_no_ood"
            basis_path = os.path.join(folder_path_contrasting, class_combination_folder, "nmf", "nmf_basis.pt")

            if not os.path.exists(basis_path):
                # contrasting has not been done for those two classes yet, do it now
                # (precomputation is not really possible in this case, due to having 1000*1000 combinations, each one taking some time)
                do_nmf_decomp(model_name, layer_name, int(class_idx), int(max_class))

            activation_vectors = torch.load(basis_path).to(DEVICE)
            pred_class, max_class, max_pred = get_pred(class_idx, activations, image_dataset, activation_vectors)

            contrasting_class_preds.append(pred_class)
            if max_class == class_idx:
                contrasting_matches_desired_pred.append(torch.tensor(1.0))
            else:
                contrasting_matches_desired_pred.append(torch.tensor(0.0))

            """
        #break





    class_preds = torch.stack(class_preds, dim=0)
    matches_desired_pred = torch.stack(matches_desired_pred, dim=0)

    out_save_path = os.path.join(OUTPUT_PATH, "sanity_check", model_name + layer_name)
    Path(out_save_path).mkdir(parents=True, exist_ok=True)

    print("average pred: {}".format(torch.mean(class_preds)))
    print("average matches_desired_pred: {}".format(torch.mean(matches_desired_pred)))

    if save_output:
        if not exclude_target_class_from_patches:
            if use_average_pooling:
                torch.save(class_preds, os.path.join(out_save_path, "nmf_comp_check_predictions_for_class_avg_pool.pt"))
                torch.save(matches_desired_pred, os.path.join(out_save_path, "nmf_comp_check_matches_desired_pred_avg_pool.pt"))
            else:
                torch.save(class_preds, os.path.join(out_save_path, "nmf_comp_check_predictions_for_class.pt"))
                torch.save(matches_desired_pred, os.path.join(out_save_path, "nmf_comp_check_matches_desired_pred.pt"))
        else:
            if use_average_pooling:
                torch.save(class_preds, os.path.join(out_save_path, "nmf_comp_check_exclude_target_predictions_for_class_avg_pool.pt"))
                torch.save(matches_desired_pred, os.path.join(out_save_path, "nmf_comp_check_exclude_target_matches_desired_pred_avg_pool.pt"))
            else:
                torch.save(class_preds, os.path.join(out_save_path, "nmf_comp_check_exclude_target_predictions_for_class.pt"))
                torch.save(matches_desired_pred, os.path.join(out_save_path, "nmf_comp_check_exclude_target_matches_desired_pred.pt"))

    """
    contrasting_class_preds = torch.stack(contrasting_class_preds, dim=0)
    contrasting_matches_desired_pred = torch.stack(contrasting_matches_desired_pred, dim=0)

    print("average contrasting_class_preds: {}".format(torch.mean(contrasting_class_preds)))
    print("average contrasting_matches_desired_pred: {}".format(torch.mean(contrasting_matches_desired_pred)))
    """



#model_name="resnet50_robust"
model_name="resnet50"
layer_name="_layer3[5]"

model = load_model(model_name).eval().to(DEVICE)
layer = model.layer3[5]

run_test(exclude_target_class_from_patches=True, use_average_pooling=True, save_output=True)
run_test(exclude_target_class_from_patches=False, use_average_pooling=True, save_output=True)


model_name="resnet50"
layer_name="_layer4[2]"

model = load_model(model_name).eval().to(DEVICE)
layer = model.layer4[2]

run_test(exclude_target_class_from_patches=True, use_average_pooling=True, save_output=True)
run_test(exclude_target_class_from_patches=False, use_average_pooling=True, save_output=True)





model_name="resnet34"
layer_name="_layer3[5]"

model = load_model(model_name).eval().to(DEVICE)
layer = model.layer3[5]

run_test(exclude_target_class_from_patches=True, use_average_pooling=True, save_output=True)
run_test(exclude_target_class_from_patches=False, use_average_pooling=True, save_output=True)


model_name="resnet34"
layer_name="_layer4[2]"

model = load_model(model_name).eval().to(DEVICE)
layer = model.layer4[2]

run_test(exclude_target_class_from_patches=True, use_average_pooling=True, save_output=True)
run_test(exclude_target_class_from_patches=False, use_average_pooling=True, save_output=True)





model_name="resnet50_robust"
layer_name="_layer3[5]"

model = load_model(model_name).eval().to(DEVICE)
layer = model.layer3[5]

run_test(exclude_target_class_from_patches=True, use_average_pooling=True, save_output=True)
run_test(exclude_target_class_from_patches=False, use_average_pooling=True, save_output=True)

model_name="resnet50_robust"
layer_name="_layer4[2]"

model = load_model(model_name).eval().to(DEVICE)
layer = model.layer4[2]

run_test(exclude_target_class_from_patches=True, use_average_pooling=True, save_output=True)
run_test(exclude_target_class_from_patches=False, use_average_pooling=True, save_output=True)



