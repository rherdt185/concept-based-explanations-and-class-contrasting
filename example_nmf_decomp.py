from torch.utils.data import DataLoader, TensorDataset

from tqdm import tqdm

from core.core import nmf_attribution_whole_ds_decomp
from core.digipath_utils import collect_dataset, get_model, collect_patches_from_prediction, AttributionLimitToTargetClassWrapper
from core.config import DEVICE, OUTPUT_PATH



for class_idx in tqdm(range(46), ascii=True):
    if class_idx > 0:
        dataset = collect_dataset(target_class=class_idx)
        if dataset is not None:
            if len(dataset) > 0:
                dataloader = DataLoader(dataset, shuffle=True, batch_size=1, num_workers=1)

                model = get_model("version_299").eval().to(DEVICE)
                layer = getattr(model.model.encoder.down_blocks, 'down block 3').resnet_blocks[2]

                wrapped_model = AttributionLimitToTargetClassWrapper(model, target_class=class_idx)

                patches = collect_patches_from_prediction(dataloader, model, target_channel=class_idx, num_to_sample=60)
                if len(patches) > 0:
                    dataset = TensorDataset(patches)


                    save_folder_path = OUTPUT_PATH + "version_299_layer3_nmf_decomp"

                    nmf_attribution_whole_ds_decomp(dataset, target_channel=class_idx, model=wrapped_model, layer=layer,
                                                    save_folder_path=save_folder_path, n_components=6,
                                                    save_images=False, batch_size=1)




