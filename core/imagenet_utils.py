
import torch
import numpy as np
import os
import csv
import torch
import torchvision
import copy

from tqdm import tqdm

from pathlib import Path

from torchvision.datasets.folder import default_loader

import torchvision
from torchvision.models import resnet50, resnet34, vit_b_32, vit_b_16


from core.config import PATH_TO_IMAGENET_TRAIN, PATH_TO_IMAGENET_VAL, PATH_VAL_SOLUTIONS, OUTPUT_PATH

DATA_RESIZE_SIZE = 224 #256 #480
DATA_CROP_SIZE = 224 #224 #480





def load_model(model_str):
    if model_str == "resnet50":
        return resnet50(pretrained=True).to("cuda").eval()
    elif model_str == "resnet34":
        return resnet34(pretrained=True).to("cuda").eval()
    elif model_str == "resnet50_robust":
        classifier_model = resnet50(pretrained=False)

        checkpoint = torch.load("/home/digipath2/projects/xai/digipath_xai_fast_api/models/ImageNet.pt")

        state_dict_path = 'model'
        if not ('model' in checkpoint):
            state_dict_path = 'state_dict'

        new_state_dict = {}
        for key in checkpoint['model'].keys():
            if not "attacker" in key:
                if 'model' in key:
                    new_key = key[13:]
                    #print(new_key)
                    new_state_dict[new_key] = checkpoint['model'][key]

        classifier_model.load_state_dict(new_state_dict)

        return classifier_model.eval().to("cuda")
    elif ((model_str == "vit_b_16_dino") or (model_str == "vit_b_16_dino_ignore_last_token")):
        #classifier_model = vit_b_16(pretrained=False)
        classifier_model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')

        #state_dict = torch.load("models/dino_vitbase16_pretrain.pth")

        #print(state_dict.keys())
        #raise RuntimeError

        #classifier_model.load_state_dict(state_dict)
        #raise RuntimeError
        return classifier_model.eval().to("cuda")
    elif model_str == "vit_b_8_dino":
        classifier_model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8')
        return classifier_model.eval().to("cuda")
    elif model_str == "vit_b_32":
        return vit_b_32(pretrained=True).eval()#.to("cuda").eval()
    elif model_str == "vit_b_16":
        return vit_b_16(pretrained=True).eval()#.to("cuda").eval()




class ImageNetValDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path_val, transform=None, transform_original=None, return_original_image=False):
        super().__init__()
        self.dataset_path = dataset_path_val
        self.transform = transform
        self.transform_original = transform_original
        self.return_original_image = return_original_image

        dataset_path_train = PATH_TO_IMAGENET_TRAIN#"/localdata/xai_derma/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train/"

        wnid_to_class = {}
        i = 0
        for folder_name in sorted(os.listdir(dataset_path_train)):
            wnid_to_class[folder_name] = i
            i += 1
        val_solutions_path = PATH_VAL_SOLUTIONS#"/localdata/xai_derma/imagenet-object-localization-challenge/LOC_val_solution.csv"

        self.image_id_to_class = {}
        with open(val_solutions_path, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=" ")
            for i, line in enumerate(reader):
                if i == 0:
                    continue
                image_id, wnid = line[0].split(",")
                self.image_id_to_class[image_id] = wnid_to_class[wnid]

        self.files = os.listdir(self.dataset_path)


    def __len__(self):
        #print("len dataset: {}".format(len(os.listdir(self.dataset_path))))
        return len(self.files)


    def __getitem__(self, index):
        filepath = self.files[index]
        class_idx = self.image_id_to_class[filepath.split(".")[0]]

        sample = default_loader(os.path.join(self.dataset_path, filepath))
        original_sample = copy.deepcopy(sample)
        if self.transform is not None:
            sample = self.transform(sample)
            #return self.transform(sample), class_idx

        if self.return_original_image:
            #print("return original sample")
            original_sample = self.transform_original(original_sample)
            return sample, class_idx, original_sample

        #print("do not return original sample")
        return sample, class_idx



def get_dataset(return_original_sample=False, use_train_ds=True):
    transform = [
        torchvision.transforms.Resize(DATA_RESIZE_SIZE),
        torchvision.transforms.CenterCrop(DATA_CROP_SIZE),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
    transform = torchvision.transforms.Compose(transform)


    transform_original_sample = [
        torchvision.transforms.Resize(DATA_RESIZE_SIZE),
        torchvision.transforms.CenterCrop(DATA_CROP_SIZE),
        torchvision.transforms.ToTensor(),
    ]
    transform_original_sample = torchvision.transforms.Compose(transform_original_sample)


    if return_original_sample:
        dataset = ImageNetValDataset(transform=transform,
                                    dataset_path_val=PATH_TO_IMAGENET_VAL,
                                    return_original_image=True,
                                    transform_original=transform_original_sample)
    else:
        if use_train_ds:
            dataset = torchvision.datasets.ImageFolder(PATH_TO_IMAGENET_TRAIN, transform=transform)
        else:
            dataset = ImageNetValDataset(transform=transform,
                                    dataset_path_val=PATH_TO_IMAGENET_VAL)

    return dataset



def get_dataset_excluding_class(model_name, class_idx):
    dataset = get_dataset(return_original_sample=False, use_train_ds=False)

    prediction_classes = np.load(os.path.join(OUTPUT_PATH, "predictions_imagenet/" + model_name + "_val.npy"))

    indices = np.where(prediction_classes != class_idx)[0]
    dataset = torch.utils.data.Subset(dataset, indices)
    #print(np.where(prediction_classes == class_idx)[0])

    return dataset




def get_dataset_for_class(model_name, class_idx):
    dataset = get_dataset(return_original_sample=False, use_train_ds=False)

    prediction_classes = np.load(os.path.join(OUTPUT_PATH, "predictions_imagenet/" + model_name + "_val.npy"))

    indices = np.where(prediction_classes == class_idx)[0]
    dataset = torch.utils.data.Subset(dataset, indices)
    #print(np.where(prediction_classes == class_idx)[0])

    return dataset


def generate_prediction_list(model_name, dataset, save_data_append="_val"):
    model = load_model(model_name).eval().to("cuda")

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False, num_workers=8)

    predictions = []

    with torch.no_grad():
        for x, target in tqdm(dataloader, ascii=True):
            x = x.to("cuda")
            pred = torch.argmax(model(x), dim=1).cpu()
            predictions.append(pred)

    predictions = torch.cat(predictions, dim=0).numpy()

    out_folder = os.path.join(OUTPUT_PATH, "predictions_imagenet")

    Path(out_folder).mkdir(parents=True, exist_ok=True)

    np.save(os.path.join(out_folder, model_name + save_data_append + ".npy"), predictions)


