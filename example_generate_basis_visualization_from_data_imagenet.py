
import torch
import torch.nn.functional as F
import torchvision
import os
import numpy as np

from pathlib import Path
from tqdm import tqdm

from torch.utils.data import DataLoader

from core.config import OUTPUT_PATH, DEVICE

from image_dictionary.image_dictionary_generation import sample_closest_images
from image_dictionary.dataset_utils import CroppedDataset

from core.imagenet_utils import get_dataset, load_model
from core.core import access_activations_forward_hook





def generate_data(model, layer, dataloader, kernel_size=2, stride=2, use_average_pooling=True):
    all_activations = []

    with torch.no_grad():
        for images, _ in tqdm(dataloader, ascii=True):
            images = images.to(DEVICE)
            activations = access_activations_forward_hook([images], model, layer)
            if use_average_pooling:
                activations = F.avg_pool2d(activations, kernel_size=kernel_size, stride=stride)
            activations = activations.cpu()
            all_activations.append(activations)
            #break

    all_activations = torch.cat(all_activations, dim=0)
    print("all activations shape: {}".format(all_activations.shape))
    return all_activations


def generate_data_including_prediction(model, layer, dataloader, kernel_size=2, stride=2, use_average_pooling=True):
    all_activations = []
    all_predictions = []

    for images, _ in tqdm(dataloader, ascii=True):
        images = images.to(DEVICE)
        with torch.no_grad():
            activations = access_activations_forward_hook([images], model, layer)
            if use_average_pooling:
                activations = F.avg_pool2d(activations, kernel_size=kernel_size, stride=stride)
            activations = activations.cpu()
            preds = torch.argmax(F.softmax(model(images), dim=1), dim=1).cpu()

        all_activations.append(activations)
        all_predictions.append(preds)
        #break

    all_activations = torch.cat(all_activations, dim=0)
    all_predictions = torch.cat(all_predictions, dim=0)
    print("all activations shape: {}".format(all_activations.shape))
    return all_activations, all_predictions


def generate_data_exclude_class(model, layer, dataloader, kernel_size=2, class_to_exclude=None):
    all_activations = []
    all_images = []

    idx = 0
    for images, targets in tqdm(dataloader, ascii=True):
        if class_to_exclude is not None:
            imgs = []
            for i, target in enumerate(targets):
                if target != class_to_exclude:
                    imgs.append(images[i])

            images = torch.stack(imgs, dim=0)

        images = images.to(DEVICE)

        if class_to_exclude is not None:
            imgs = []
            with torch.no_grad():
                preds = F.softmax(model(images), dim=1)
                for i, pred in enumerate(preds):
                    if torch.argmax(pred) != class_to_exclude:
                        imgs.append(images[i])

            images = torch.stack(imgs, dim=0)

        activations = access_activations_forward_hook([images], model, layer)
        activations = F.avg_pool2d(activations, kernel_size=kernel_size, stride=2)
        all_activations.append(activations)
        all_images.append(images)
        idx += 1
        if idx >= 2:
            break

    all_activations = torch.cat(all_activations, dim=0)
    all_images = torch.cat(all_images, dim=0)
    print("all activations shape: {}".format(all_activations.shape))
    return all_activations, all_images


def sample_closest_image_patches(activation_vector, activation_data, image_dataset, n_to_sample=8, is_image_dataset_list=False,
                                 return_full_imgs_as_well=False):
    b, c, h, w = activation_data.shape
    activation_data = activation_data.permute(0, 2, 3, 1).reshape(-1, c)

    activation_vector = activation_vector.unsqueeze(dim=0)

    #print("len activation data: {}".format(len(activation_data)))
    step_size = int(b/5)
    cosine_similarity = []
    for i in range(0, len(activation_data), step_size):
        cossim = F.cosine_similarity(activation_data[i:(i+step_size)].to(DEVICE), activation_vector).cpu()
        #print(activation_data[i:(i+step_size)].shape)
        cosine_similarity.append(cossim)

    #print(len(cosine_similarity))
    cosine_similarity = torch.cat(cosine_similarity, dim=0)
    #print(cosine_similarity.shape)
    #cosine_similarity = F.cosine_similarity(activation_data, activation_vector.unsqueeze(dim=0)).cpu()
    topk_values, topk_indices = torch.topk(cosine_similarity.view(-1), n_to_sample)

    batch_indices = topk_indices // (h * w)  # Determine the batch index
    spatial_indices = topk_indices % (h * w)  # Determine the spatial index within each batch
    h_indices = spatial_indices // w
    w_indices = spatial_indices % w

    indices = torch.stack((batch_indices, h_indices, w_indices), dim=1)  # Shape (n_to_sample, 3)

    patches = []
    full_images = []

    for index in indices:
        b_idx, h_idx, w_idx = index

        if not is_image_dataset_list:
            img = image_dataset[b_idx][2]       # original unnormalized image is at index 2
        else:
            img = image_dataset[b_idx][0]
        full_images.append(img)
        start_h = int(img.shape[1] / h) * h_idx
        end_h = int(img.shape[1] / h) * (h_idx+1)

        start_w = int(img.shape[2] / w) * w_idx
        end_w = int(img.shape[2] / w) * (w_idx+1)

        patch = img[:, start_h:end_h, start_w:end_w]
        patches.append(patch)

    if return_full_imgs_as_well:
        return torch.stack(patches, dim=0), torch.stack(full_images, dim=0)
    return torch.stack(patches, dim=0)



def visualize_vectors(activation_vectors, activations, image_dataset, visualization_save_path, n_to_sample=8):

    all_closest_image_patches = []

    for activation_vector in activation_vectors:
        closest_image_patches = sample_closest_image_patches(activation_vector, activations, image_dataset, n_to_sample=n_to_sample)

        all_closest_image_patches.append(closest_image_patches)
    all_closest_image_patches = torch.cat(all_closest_image_patches, dim=2)

    torchvision.utils.save_image(all_closest_image_patches, visualization_save_path)




def visualize_contrasting(folder_path, exclude_target_class_from_patches=False):
    #activations = generate_data(model, layer, dataloader)#.to(DEVICE)
    activations_all, prediction_classes = generate_data_including_prediction(model, layer, dataloader, kernel_size=2, stride=2, use_average_pooling=True)
    prediction_classes = prediction_classes.cpu().numpy()

    img_dataset = get_dataset(return_original_sample=True, use_train_ds=False)

    for class_combination_folder in sorted(os.listdir(folder_path)):
        if exclude_target_class_from_patches:
            indices = np.where(prediction_classes != int(class_combination_folder))[0]
            image_dataset = torch.utils.data.Subset(img_dataset, indices)
            activations = activations_all[indices]
        else:
            activations = activations_all
            image_dataset = img_dataset

        print("folder path: {}".format(folder_path))
        basis_path = os.path.join(folder_path, class_combination_folder, "nmf", "nmf_basis.pt")

        # avoid crash if this is run while the data is still being generated, since the data generation creates the folder first and saves the basis later
        if os.path.exists(basis_path):
            activation_vectors = torch.load(basis_path).to(DEVICE)

            if exclude_target_class_from_patches:
                visualization_save_path = os.path.join(folder_path, class_combination_folder, "nmf", "data_vis_basis_exclude_target.jpg")
            else:
                visualization_save_path = os.path.join(folder_path, class_combination_folder, "nmf", "data_vis_basis.jpg")

            visualize_vectors(activation_vectors, activations, image_dataset, visualization_save_path, n_to_sample=8)




if __name__ == "__main__":
    dataset = get_dataset(use_train_ds=False)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=64, num_workers=8)


    model = load_model("resnet50").eval().to(DEVICE)
    layer = model.layer3[5]
    layer_name = "_layer3[5]"

    #folder_path = os.path.join(OUTPUT_PATH, "contrasting", "resnet50_robust" + layer_name)
    #visualize_contrasting(folder_path)

    folder_path = os.path.join(OUTPUT_PATH, "resnet50" + layer_name)
    visualize_contrasting(folder_path, exclude_target_class_from_patches=False)

    folder_path = os.path.join(OUTPUT_PATH, "resnet50" + layer_name)
    visualize_contrasting(folder_path, exclude_target_class_from_patches=True)





    model = load_model("resnet50").eval().to(DEVICE)
    layer = model.layer4[2]
    layer_name = "_layer4[2]"

    #folder_path = os.path.join(OUTPUT_PATH, "contrasting", "resnet50_robust" + layer_name)
    #visualize_contrasting(folder_path)

    folder_path = os.path.join(OUTPUT_PATH, "resnet50" + layer_name)
    visualize_contrasting(folder_path, exclude_target_class_from_patches=False)

    folder_path = os.path.join(OUTPUT_PATH, "resnet50" + layer_name)
    visualize_contrasting(folder_path, exclude_target_class_from_patches=True)










    model = load_model("resnet34").eval().to(DEVICE)
    layer = model.layer3[5]
    layer_name = "_layer3[5]"

    #folder_path = os.path.join(OUTPUT_PATH, "contrasting", "resnet50_robust" + layer_name)
    #visualize_contrasting(folder_path)

    folder_path = os.path.join(OUTPUT_PATH, "resnet34" + layer_name)
    visualize_contrasting(folder_path, exclude_target_class_from_patches=False)

    folder_path = os.path.join(OUTPUT_PATH, "resnet34" + layer_name)
    visualize_contrasting(folder_path, exclude_target_class_from_patches=True)





    model = load_model("resnet34").eval().to(DEVICE)
    layer = model.layer4[2]
    layer_name = "_layer4[2]"

    #folder_path = os.path.join(OUTPUT_PATH, "contrasting", "resnet50_robust" + layer_name)
    #visualize_contrasting(folder_path)

    folder_path = os.path.join(OUTPUT_PATH, "resnet34" + layer_name)
    visualize_contrasting(folder_path, exclude_target_class_from_patches=False)

    folder_path = os.path.join(OUTPUT_PATH, "resnet34" + layer_name)
    visualize_contrasting(folder_path, exclude_target_class_from_patches=True)









    model = load_model("resnet50_robust").eval().to(DEVICE)
    layer = model.layer3[5]
    layer_name = "_layer3[5]"

    #folder_path = os.path.join(OUTPUT_PATH, "contrasting", "resnet50_robust" + layer_name)
    #visualize_contrasting(folder_path)

    folder_path = os.path.join(OUTPUT_PATH, "resnet50_robust" + layer_name)
    visualize_contrasting(folder_path, exclude_target_class_from_patches=False)

    folder_path = os.path.join(OUTPUT_PATH, "resnet50_robust" + layer_name)
    visualize_contrasting(folder_path, exclude_target_class_from_patches=True)





    model = load_model("resnet50_robust").eval().to(DEVICE)
    layer = model.layer4[2]
    layer_name = "_layer4[2]"

    #folder_path = os.path.join(OUTPUT_PATH, "contrasting", "resnet50_robust" + layer_name)
    #visualize_contrasting(folder_path)

    folder_path = os.path.join(OUTPUT_PATH, "resnet50_robust" + layer_name)
    visualize_contrasting(folder_path, exclude_target_class_from_patches=False)

    folder_path = os.path.join(OUTPUT_PATH, "resnet50_robust" + layer_name)
    visualize_contrasting(folder_path, exclude_target_class_from_patches=True)

