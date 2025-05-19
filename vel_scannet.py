from a_model import *
from dataset import *

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.utils.data

import lyzKNN


def save_result(filename_, predict_, points_, points_down):
    predict_choice = predict_.data.max(1)[1]
    predict_choice = predict_choice.contiguous().cpu().data.numpy()

    p_and_l = np.hstack([points_down[:, :6], predict_choice.reshape(-1, 1)])
    p_and_l = torch.from_numpy(p_and_l).unsqueeze(0)
    points_ = torch.from_numpy(points_).unsqueeze(0).to(torch.float64)
    idx, _ = lyzKNN.knn(p_and_l[..., :3].contiguous(), points_[..., :3].contiguous(), 1)
    idx, points_ = idx.squeeze(), points_.squeeze()
    predict_choice = predict_choice[idx]

    np.savetxt(filename_, predict_choice, fmt="%d")


def main():
    device = torch.device("cuda:0")

    a = torch.load('model_result/model_scannet.pth')
    b = torch.load('model_result/model_head_scannet.pth')

    # collect_fun = SparseCollation(voxel_size=0.07)

    # test_file_dir = '/home/lee/Documents/dataset/scannet/Ours/test'
    test_file_dir = "/home/lee/Documents/dataset/scannet/scannet_ptv3/test"
    dataset_path = []
    for i in os.listdir(test_file_dir):
        dataset_path.append(os.path.join(test_file_dir, i))

    model = OursModel(6, is_cat=True).to(device)
    model_head = SegmentationClassifierHead(32, 50).to(device)

    model.load_state_dict(a)
    model_head.load_state_dict(b)

    model.eval()
    model_head.eval()

    for data in tqdm(dataset_path, total=len(dataset_path)):
        file_name = '/home/lee/Documents/dataset/scannet/Ours/result/'
        file_name += data.split('/')[-1][: -3]
        file_name += 'txt'

        # points_set = np.load(data)
        # points = points_set[:, :6]

        points_set = torch.load(data)
        points = np.hstack([points_set['coord'] - np.mean(points_set['coord'], 0), points_set['color'] / 255])
        points = torch.from_numpy(points).contiguous()
        points = np.array(points)
        coord, feat, use_index = ME.utils.sparse_quantize(points[:, :3], points,
                                                          return_index=True, ignore_label=-1,
                                                          quantization_size=0.07)
        coord, feat = [coord], [feat]
        coord = ME.utils.batched_coordinates(coord)
        feat = torch.from_numpy(np.concatenate(feat, 0)).float()
        input_data = ME.SparseTensor(feat.float(), coord, device=device)

        points_down = points[use_index]

        input_data = model(input_data)
        predi_ = model_head(input_data)

        # save_result(file_name, predi_, points, points_down)


if __name__ == '__main__':
    main()
