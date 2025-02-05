
from core.imagenet_utils import get_dataset, generate_prediction_list


dataset = get_dataset(return_original_sample=False, use_train_ds=False)
generate_prediction_list("resnet50", dataset, save_data_append="_val")

dataset = get_dataset(return_original_sample=False, use_train_ds=False)
generate_prediction_list("resnet34", dataset, save_data_append="_val")

dataset = get_dataset(return_original_sample=False, use_train_ds=False)
generate_prediction_list("resnet50_robust", dataset, save_data_append="_val")


