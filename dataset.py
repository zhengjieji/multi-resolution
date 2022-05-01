import os, torch
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms

# Global
np.random.seed(0)
    
class MultiResolutionDataset(data.Dataset):
    def __init__(self, depth, depth_lq, sidescan):
        self.depth = depth
        self.depth_lq = depth_lq
        self.sidescan = sidescan

    def __getitem__(self, index):
        depth_image = np.load(self.depth[index]).reshape(1, 256, 256)
        depth_lq_image = np.load(self.depth_lq[index]).reshape(1, 256, 256) 
        sidescan_image = np.load(self.sidescan[index]).reshape(1, 256, 256) 
        return depth_image, depth_lq_image, sidescan_image
        
    def __len__(self):
        return len(self.depth)


def load_split(root, split, lq_scale):
    split_path = root + '/' + split
    depth = []
    depth_lq = []
    sidescan = []
    folders = os.listdir(split_path)

    if '.DS_Store' in folders:
        folders.remove('.DS_Store')

    for i in range(len(folders)):
        sample_path = split_path + '/' + str(i) 
        depth.append(sample_path + '/d' + '.npy')
        depth_lq.append(sample_path + '/d_' + str(lq_scale) + '.npy')
        sidescan.append(sample_path + '/s' + '.npy')

    depth = np.array(depth)
    depth_lq = np.array(depth_lq)
    sidescan = np.array(sidescan)
    return depth, depth_lq, sidescan

def get_dataset(root, split, lq_scale):
    depth, depth_lq, sidescan = load_split(root, split, lq_scale)
    dataset = MultiResolutionDataset(depth, depth_lq, sidescan)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
    return dataset


if __name__ == "__main__":
    ROOT = os.getcwd()
    data_root = ROOT + '/sss2depth_split'
    train_dataloader = get_dataloader(data_root, 'train', 2)
