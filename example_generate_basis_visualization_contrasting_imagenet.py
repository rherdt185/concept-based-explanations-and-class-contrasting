
import torch
import torchvision
import os

from pathlib import Path

import torch.nn.functional as F

from core.digipath_utils import load_decoder
from core.imagenet_utils import load_model
from core.visualization import generate_gan_visualizations

from core.optimization_vis import optim_gan, dot_cossim_loss, LossObject_Target, OptimGoal_ForwardHook

from core.utils import access_activations_forward_hook

from core.config import OUTPUT_PATH, DEVICE

from decoder.core import FeatureExtractorForwardWrapper
from torchvision.models.feature_extraction import create_feature_extractor


classifier = load_model("resnet50_robust").eval().to(DEVICE)

layer_name = "_layer3[5]"

layer = "layer3.5"

return_nodes = {
    # node_name: user-specified key for output dict
    layer: 'out',
}

model_from = create_feature_extractor(classifier, return_nodes).eval()

decoder = load_decoder("resnet50_robust")

decoder.feature_extractor = model_from
feature_extractor = decoder.feature_extractor

model = FeatureExtractorForwardWrapper(feature_extractor)

#folder_path = "/home/digipath2/projects/xai/digipath_xai_fast_api/highest_pixels/version_299_nmf_full_ds_decomp/"
#folder_path = os.path.join(OUTPUT_PATH, "version_299_layer3_nmf_decomp")

#folder_path = os.path.join(OUTPUT_PATH, "contrasting", "resnet50_robust" + layer_name)
folder_path = os.path.join(OUTPUT_PATH, "resnet50_robust" + layer_name)

for class_combination_folder in os.listdir(folder_path):
    basis_path = os.path.join(folder_path, class_combination_folder, "nmf", "nmf_basis.pt")
    activation_vectors = torch.load(basis_path).to(DEVICE)*50.0#2.0
    print("generate visualizations...")

    b_data = torch.randn((activation_vectors.shape[0], 3, 224, 224)).to("cuda")

    #loss_function = dot_cossim_loss
    loss_function = lambda x, target: -F.l1_loss(torch.mean(x, dim=[2, 3]), target)
    loss_object = LossObject_Target(loss_function, target=activation_vectors)
    optim_goal = OptimGoal_ForwardHook(model.identity_layer_for_output_hook, loss_object, weight=-1.0)

    decoder_latent_hook = decoder.decoder.decoder.up_blocks[2]  # default is 2
    initial_gan_activations = access_activations_forward_hook([activation_vectors.float().to(DEVICE)], decoder.forward, decoder_latent_hook)
    visualizations = optim_gan(initial_gan_activations, decoder.forward_no_inputs, decoder_latent_hook,
                            model, [optim_goal], from_single_vector=False, num_steps=64,
                            normalization_preprocessor=torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))).detach().to("cpu")



    #visualizations_layer_0, visualizations_layer_3 = generate_gan_visualizations(activation_vectors, decoder, feature_extractor)

    torchvision.utils.save_image(visualizations, os.path.join(folder_path, class_combination_folder, "nmf", "basis_vis_"+str(1)+".jpg"))
    #torchvision.utils.save_image(visualizations_layer_3, os.path.join(folder_path, class_combination_folder, "nmf", "basis_3_"+str(1)+".jpg"))








