import os
import numpy as np

from torch.utils.data import Dataset
import torch

import MinkowskiEngine as ME


class SparseCollation:
    def __init__(self, voxel_size):
        self.voxel_size = voxel_size

    def __call__(self, list_data):
        p_coord, p_feats, p_label = [], [], []

        points_set, seg_labels = list(zip(*list_data))
        for points_i, labels_i in zip(points_set, seg_labels):
            coord, feats, labels = ME.utils.sparse_quantize(points_i[:, :3], points_i, labels_i,
                                                            ignore_label=0, quantization_size=self.voxel_size)
            p_coord.append(coord)
            p_feats.append(feats)
            p_label.append(labels)

        coord_batch = ME.utils.batched_coordinates(p_coord)
        feats_batch = torch.from_numpy(np.concatenate(p_feats, 0)).float()
        labels_batch = torch.from_numpy(np.concatenate(p_label, 0))

        return coord_batch, feats_batch, labels_batch


class S3DIS(Dataset):
    def __init__(self, split='train', data_root='/home/lee/Documents/dataset/s3dis/stanford_indoor3d', test_area=5):
        super().__init__()
        data_list = sorted(os.listdir(data_root))
        data_list = [item[:-4] for item in data_list if 'Area_' in item]
        if split == 'train':
            self.data_list = [item for item in data_list if not 'Area_{}'.format(test_area) in item]
        else:
            self.data_list = [item for item in data_list if 'Area_{}'.format(test_area) in item]

        self.data_root = data_root
        # for item in self.data_list:
        #     if not os.path.exists("/dev/shm/{}".format(item)):
        #         data_path = os.path.join(data_root, item + '.npy')
        #         data = np.load(data_path)  # xyzrgbl, N*7
        #         sa_create("shm://{}".format(item), data)
        self.data_idx = np.arange(len(self.data_list))
        print("Totally {} samples in {} set.".format(len(self.data_idx), split))

    def __getitem__(self, idx):
        data_idx = self.data_idx[idx % len(self.data_idx)]

        item = self.data_list[data_idx]
        data_path = os.path.join(self.data_root, item + '.npy')
        data = np.load(data_path)

        coord, label = data[:, 0:6], data[:, 6]

        return coord, label

    def __len__(self):
        return len(self.data_idx)
