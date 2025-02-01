import os

from core.separate_features import train_linear_classifiers, load_activation_vecs_from_components
from core.config import OUTPUT_PATH

train_linear_classifiers(activation_vecs_folder=os.path.join(OUTPUT_PATH, "version_299_layer3_nmf_decomp"),
                         output_folder=os.path.join(OUTPUT_PATH, "hyperplanes_from_nmf_components", "version_299"),
                         features_list=[i for i in range(46)],
                         load_function=load_activation_vecs_from_components)


