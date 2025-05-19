import os
import open3d as o3d
import numpy as np
from tqdm import tqdm


def random_sample(arr_, ratio):
    num_p = arr_.shape[0]
    if ratio < 1:
        arr_i = np.random.choice(num_p, np.floor(num_p * ratio).astype(int), replace=False)
    else:
        arr_i = np.random.choice(num_p, ratio, replace=False)

    arr_new = arr_[arr_i, :]

    return arr_new


def generate_segmentation_dataset(path_, save_path, generate_num=100):
    if not os.path.exists(os.path.join(save_path, 'segmentDataset')):
        os.mkdir(os.path.join(save_path, 'segmentDataset'))

    sub_folder = os.listdir(path_)
    for num_i, i in tqdm(enumerate(sub_folder), total=len(sub_folder)):
        if not os.path.exists(os.path.join(save_path, 'segmentDataset', i)):
            os.mkdir(os.path.join(save_path, 'segmentDataset', i))
            num_file = 0
        else:
            num_file = len(os.listdir(os.path.join(save_path, 'segmentDataset', i)))

        sub_sub_folder = os.listdir(os.path.join(path_, i))
        station_pc = []
        for j in sub_sub_folder:
            pcd_file = os.path.join(path_, i, j)
            pcd_origin = o3d.t.io.read_point_cloud(pcd_file)
            if len(pcd_origin.point.positions) > 8e+7:
                pcd_origin = pcd_origin.random_down_sample(sampling_ratio=0.5)

            pcd_xyz_ = pcd_origin.point.positions.numpy()
            pcd_rgb_ = pcd_origin.point.colors.numpy()
            if 'bridge' in j.lower():
                pcd_label_ = np.zeros((pcd_xyz_.shape[0], 1))
                pcd_ = np.hstack((pcd_xyz_, pcd_rgb_, pcd_label_))
                pcd_ = random_sample(pcd_, 0.5)
            elif 'fixture' in j.lower():
                pcd_label_ = np.ones((pcd_xyz_.shape[0], 1))
                pcd_ = np.hstack((pcd_xyz_, pcd_rgb_, pcd_label_))
                pcd_ = random_sample(pcd_, 0.5)
            elif 'safe' in j.lower():
                pcd_label_ = np.ones((pcd_xyz_.shape[0], 1)) * 2
                pcd_ = np.hstack((pcd_xyz_, pcd_rgb_, pcd_label_))
                pcd_ = random_sample(pcd_, 0.5)
            elif 'other' in j.lower():
                pcd_label_ = np.ones((pcd_xyz_.shape[0], 1)) * 3
                pcd_ = np.hstack((pcd_xyz_, pcd_rgb_, pcd_label_))
                pcd_ = random_sample(pcd_, 0.5)
            elif 'robot' in j.lower():
                pcd_label_ = np.ones((pcd_xyz_.shape[0], 1)) * 4
                pcd_ = np.hstack((pcd_xyz_, pcd_rgb_, pcd_label_))
                pcd_ = random_sample(pcd_, 0.5)
            elif 'ground' in j.lower():
                pcd_label_ = np.ones((pcd_xyz_.shape[0], 1)) * 5
                pcd_ = np.hstack((pcd_xyz_, pcd_rgb_, pcd_label_))
                pcd_ = random_sample(pcd_, 0.5)

            station_pc.append(pcd_)

        del pcd_origin, pcd_xyz_, pcd_rgb_, pcd_label_, pcd_
        station_pc = np.concatenate(station_pc, axis=0)
        for file_i in range(generate_num):
            station_pc_rs = random_sample(station_pc, 180000)
            rs_mean = np.mean(station_pc_rs[:, 0: 3], axis=0)
            station_pc_rs[:, 0: 3] = station_pc_rs[:, 0: 3] - rs_mean

            rotate_angle = np.random.rand() * np.pi * 2
            rotate_z = np.array([[np.cos(rotate_angle), np.sin(rotate_angle), 0],
                                 [-np.sin(rotate_angle), np.cos(rotate_angle), 0], [0, 0, 1]])
            station_pc_rs[:, 0: 3] = station_pc_rs[:, 0: 3] @ rotate_z.T

            station_pc_rs.tofile(os.path.join(save_path, 'segmentDataset', i, '{}.bin'.format(num_file + file_i + 1)))


if __name__ == '__main__':
    generate_segmentation_dataset('/home/lee/Documents/dataset/digital_factory/RawStationPC',
                                  '/home/lee/Documents/dataset/digital_factory/allClassSegmentDataset',
                                  generate_num=100)
