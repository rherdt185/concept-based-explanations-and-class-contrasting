# Use a 1536x1536 dataset and cut it into 512x512 patches
# Make it messy in the dataset, to keep it clean in the core code.
# Ideally the dataset should be generated at 512x512 and then used
# in the core code as is, without wrapping it into a CroppedDataset


from torch.utils.data import Dataset


class CroppedDataset(Dataset):
    def __init__(self, digipath_dataset):
        self.digipath_dataset = digipath_dataset
        self.num_datapoints_digipath_dataset = len(self.digipath_dataset)

    def __getitem__(self, index):
        remapped_index = int(index / 9)# + index % 9
        #print("index: {}, remapped_index: {}, len digipath ds: {}, len this ds: {}".format(index, remapped_index, len(self.digipath_dataset), len(self)))
        data = self.digipath_dataset[remapped_index]
        patch = data[0]

        height_index = int((index % 9) / 3)
        width_index = (index % 9) % 3

        return patch[:, (height_index*512):((height_index+1)*512), (width_index*512):((width_index+1)*512)], 0.0

    def __len__(self):
        return self.num_datapoints_digipath_dataset*9
