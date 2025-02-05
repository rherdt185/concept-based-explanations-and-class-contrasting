
import torch
import torch.nn.functional as F
from tqdm import tqdm

from core.config import DEVICE
from core.utils import access_activations_forward_hook



def generate_mean_activation_data(dataloader, model, layer):
    with torch.no_grad():
        activations_data = []
        for data in tqdm(dataloader, ascii=True):
            images = data[0].to(DEVICE)

            activations = access_activations_forward_hook([images], model, layer)
            activations_data.append(torch.mean(activations, dim=[2, 3]))

        activations_data = torch.cat(activations_data, dim=0)

        return activations_data


def sample_closest_images(dataset, dataset_activation_vectors, activation_vectors_for_vis,
                          n_images_to_sample=8, min_distance_between_sampled_images=0):

    all_images = []

    for activation_vector in activation_vectors_for_vis:
        images = []
        cossim = F.cosine_similarity(dataset_activation_vectors, activation_vector.unsqueeze(dim=0), dim=1)
        sorted_cossim, indices = torch.sort(cossim, descending=True)

        #for k in range(n_images_to_sample):
        sorted_index = 0
        sampled_dataset_indice = []
        while len(images) < n_images_to_sample:
            dataset_index = indices[sorted_index]
            sorted_index += 1

            closest_distance = 999999999
            for sampled_dataset_index in sampled_dataset_indice:
                closest_distance = min(abs(dataset_index - sampled_dataset_index), closest_distance)
            if closest_distance < min_distance_between_sampled_images:
                continue

            sampled_dataset_indice.append(dataset_index)
            data = dataset[dataset_index]
            image = data[0]
            images.append(image)

            if sorted_index >= len(dataset):
                break

        all_images.extend(images)

    all_images = torch.stack(all_images, dim=0)

    return all_images

