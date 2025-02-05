from torch.utils.data import DataLoader, TensorDataset

from tqdm import tqdm

from core.core import nmf_attribution_whole_ds_decomp, collect_patches
from core.imagenet_utils import get_dataset_for_class, load_model
from core.config import DEVICE, OUTPUT_PATH


def run_nmf_decomp(model, layer, save_name, model_name="resnet50_robust"):
    for class_idx in tqdm(range(1000), ascii=True):
        dataset = get_dataset_for_class(model_name=model_name, class_idx=class_idx)
        if dataset is not None:
            if len(dataset) > 0:
                dataloader = DataLoader(dataset, shuffle=True, batch_size=1, num_workers=1)

                patches = collect_patches(dataloader)
                if len(patches) > 60:
                    patches = patches[:60]
                if len(patches) > 0:
                    dataset = TensorDataset(patches)


                    save_folder_path = OUTPUT_PATH + save_name

                    nmf_attribution_whole_ds_decomp(dataset, target_channel=class_idx, model=model, layer=layer,
                                                    save_folder_path=save_folder_path, n_components=6,
                                                    save_images=False, batch_size=1)



model_name = "resnet50"
model = load_model(model_name).eval().to(DEVICE)
layer = model.layer3[5]
layer_name = "_layer3[5]"

run_nmf_decomp(model, layer, model_name=model_name, save_name=model_name+layer_name)

model_name = "resnet50"
model = load_model(model_name).eval().to(DEVICE)
layer = model.layer4[2]
layer_name = "_layer4[2]"

run_nmf_decomp(model, layer, model_name=model_name, save_name=model_name+layer_name)



model_name = "resnet34"
model = load_model(model_name).eval().to(DEVICE)
layer = model.layer3[5]
layer_name = "_layer3[5]"

run_nmf_decomp(model, layer, model_name=model_name, save_name=model_name+layer_name)

model_name = "resnet34"
model = load_model(model_name).eval().to(DEVICE)
layer = model.layer4[2]
layer_name = "_layer4[2]"

run_nmf_decomp(model, layer, model_name=model_name, save_name=model_name+layer_name)



model_name = "resnet50_robust"
model = load_model(model_name).eval().to(DEVICE)
layer = model.layer3[5]
layer_name = "_layer3[5]"

run_nmf_decomp(model, layer, model_name=model_name, save_name=model_name+layer_name)

model_name = "resnet50_robust"
model = load_model(model_name).eval().to(DEVICE)
layer = model.layer4[2]
layer_name = "_layer4[2]"

run_nmf_decomp(model, layer, model_name=model_name, save_name=model_name+layer_name)

