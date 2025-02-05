import torch
import os
#from core.config import OUTPUT_PATH

OUTPUT_PATH = "out"

model_name = "resnet50_robust"
#model_name = "resnet50"
layer_name = "_layer4[2]"


out_save_path = os.path.join(OUTPUT_PATH, "sanity_check", model_name + layer_name)
#class_preds = torch.load(os.path.join(out_save_path, "nmf_comp_check_exclude_target_predictions_for_class_avg_pool.pt"))
#matches_desired_pred = torch.load(os.path.join(out_save_path, "nmf_comp_check_exclude_target_matches_desired_pred_avg_pool.pt"))
class_preds = torch.load(os.path.join(out_save_path, "nmf_comp_check_predictions_for_class_avg_pool.pt"))
matches_desired_pred = torch.load(os.path.join(out_save_path, "nmf_comp_check_matches_desired_pred_avg_pool.pt"))


print(torch.mean(class_preds))
print(torch.std(class_preds))
print(torch.mean(matches_desired_pred))

above_point_one_pred = torch.where(class_preds > 0.1, 1.0, 0.0)
print(torch.mean(above_point_one_pred))

above_point_two_pred = torch.where(class_preds > 0.2, 1.0, 0.0)
print(torch.mean(above_point_two_pred))

above_point_five_pred = torch.where(class_preds > 0.5, 1.0, 0.0)
print(torch.mean(above_point_five_pred))


#for i in range(100):
#    print(i, class_preds[i])