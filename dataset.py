import os
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans

from torch.utils.data import Dataset, DataLoader
import torch

import MinkowskiEngine as ME


def cluster_pc(array, n_cluster=50):
    labels = KMeans(n_clusters=n_cluster, n_init='auto').fit_predict(array)
    labels = labels.reshape(-1, 1)

    return np.concatenate([array, labels], axis=1)


def point_set_to_coord_feats(point_set, labels, resolution, num_points, deterministic=False):
    p_feats = point_set.copy()
    p_coord = np.round(point_set[:, :3] / resolution)
    p_coord -= p_coord.min(0, keepdims=1)

    _, mapping = ME.utils.sparse_quantize(coordinates=p_coord, return_index=True)
    if len(mapping) > num_points:
        if deterministic:
            np.random.seed(42)

        mapping = np.random.choice(mapping, num_points, replace=False)

    return p_coord[mapping], p_feats[mapping], labels[mapping]


def overlap_clusters(cluster_i, cluster_j, min_cluster_point=20):
    # get unique labels from pcd_i and pcd_j
    unique_i = np.unique(cluster_i)
    unique_j = np.unique(cluster_j)

    # get labels present on both pcd (intersection)
    unique_ij = np.intersect1d(unique_i, unique_j)

    # also remove clusters with few points
    for cluster in unique_ij.copy():
        ind_i = np.where(cluster_i == cluster)
        ind_j = np.where(cluster_j == cluster)

        if len(ind_i[0]) < min_cluster_point or len(ind_j[0]) < min_cluster_point:
            unique_ij = np.delete(unique_ij, unique_ij == cluster)

    # labels not intersecting both pcd are assigned as -1 (unlabeled)
    cluster_i[np.in1d(cluster_i, unique_ij, invert=True)] = -1
    cluster_j[np.in1d(cluster_j, unique_ij, invert=True)] = -1

    return cluster_i, cluster_j


class SparseAugmentedCollation:
    def __init__(self, resolution, num_points=80000, segment_contrast=True):
        self.resolution = resolution
        self.num_points = num_points
        self.segment_contrast = segment_contrast

    def __call__(self, list_data):
        points_i, points_j = list(zip(*list_data))

        points_i = np.asarray(points_i, dtype=object)
        points_j = np.asarray(points_j, dtype=object)

        pi_feats = []
        pi_coord = []
        pi_cluster = []

        pj_feats = []
        pj_coord = []
        pj_cluster = []

        for pi, pj in zip(points_i, points_j):
            coord_pi, feats_pi, cluster_pi = point_set_to_coord_feats(pi[:, :-1], pi[:, -1],
                                                                      self.resolution, self.num_points)
            pi_coord.append(coord_pi)
            pi_feats.append(feats_pi)

            coord_pj, feats_pj, cluster_pj = point_set_to_coord_feats(pj[:, :-1], pj[:, -1],
                                                                      self.resolution, self.num_points)
            pj_coord.append(coord_pj)
            pj_feats.append(feats_pj)

            cluster_pi, cluster_pj = overlap_clusters(cluster_pi, cluster_pj)

            if self.segment_contrast:
                pi_cluster.append(cluster_pi)
                pj_cluster.append(cluster_pj)

        pi_feats = np.asarray(pi_feats, dtype=object)
        pi_coord = np.asarray(pi_coord, dtype=object)

        pj_feats = np.asarray(pj_feats, dtype=object)
        pj_coord = np.asarray(pj_coord, dtype=object)

        segment_i = np.asarray(pi_cluster, dtype=object)
        segment_j = np.asarray(pj_cluster, dtype=object)

        # if not segment_contrast segment_i and segment_j will be an empty list
        return (pi_coord, pi_feats, segment_i), (pj_coord, pj_feats, segment_j)


def rotate_point_cloud(batch_data):
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, -sinval, 0],
                                    [sinval, cosval, 0],
                                    [0, 0, 1]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)

    return rotated_data


def rotate_perturbation_point_cloud(batch_data, angle_sigma=0.06, angle_clip=0.18):
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        angles = np.clip(angle_sigma * np.random.randn(3), -angle_clip, angle_clip)
        rx = np.array([[1, 0, 0],
                       [0, np.cos(angles[0]), -np.sin(angles[0])],
                       [0, np.sin(angles[0]), np.cos(angles[0])]])
        ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                       [0, 1, 0],
                       [-np.sin(angles[1]), 0, np.cos(angles[1])]])
        rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                       [np.sin(angles[2]), np.cos(angles[2]), 0],
                       [0, 0, 1]])
        r = np.dot(rz, np.dot(ry, rx))
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), r)
    return rotated_data


def random_scale_point_cloud(batch_data, scale_low=0.95, scale_high=1.05):
    batch_n, num_p, channel_n = batch_data.shape
    scales = np.random.uniform(scale_low, scale_high, batch_n)
    for batch_index in range(batch_n):
        batch_data[batch_index, :, :] *= scales[batch_index]
    return batch_data


def random_flip_point_cloud(batch_data):
    batch_, num_p, channel_ = batch_data.shape
    for batch_index in range(batch_):
        if np.random.random() > 0.5:
            batch_data[batch_index, :, 1] = -1 * batch_data[batch_index, :, 1]

    return batch_data


def random_drop_point_cloud(batch_data):
    """ Randomly drop cuboids from the point cloud.
        Input:
            BxNx3 array, original batch of point clouds
        Return:
            BxNx3 array, dropped batch of point clouds
    """
    batch_, num_p, channel_ = batch_data.shape
    new_batch_data = []
    for batch_index in range(batch_):
        range_xyz = np.max(batch_data[batch_index, :, 0:3], axis=0) - np.min(batch_data[batch_index, :, 0:3], axis=0)

        crop_range = np.random.uniform(0.1, 0.15)
        new_range = range_xyz * crop_range / 2.0
        sample_center = batch_data[batch_index, np.random.choice(len(batch_data[batch_index])), 0:3]
        max_xyz = sample_center + new_range
        min_xyz = sample_center - new_range

        upper_idx = np.sum((batch_data[batch_index, :, 0:3] < max_xyz).astype(np.int32), 1) == 3
        lower_idx = np.sum((batch_data[batch_index, :, 0:3] > min_xyz).astype(np.int32), 1) == 3

        new_pointidx = ~(upper_idx & lower_idx)
        new_batch_data.append(batch_data[batch_index, new_pointidx, :])

    return np.array(new_batch_data)


def random_drop_n_cuboids(batch_data):
    """ Randomly drop N cuboids from the point cloud.
        Input:
            BxNx3 array, original batch of point clouds
        Return:
            BxNx3 array, dropped batch of point clouds
    """
    batch_data = random_drop_point_cloud(batch_data)
    cuboids_count = 1
    while cuboids_count < 5 and np.random.uniform(0., 1.) > 0.3:
        batch_data = random_drop_point_cloud(batch_data)
        cuboids_count += 1

    return batch_data


def check_aspect_2d(crop_range, aspect_min):
    xy_aspect = np.min(crop_range[:2]) / np.max(crop_range[:2])
    return xy_aspect >= aspect_min


def random_cuboid_point_cloud(batch_data):
    """ Randomly drop cuboids from the point cloud.
        Input:
            BxNx3 array, original batch of point clouds
        Return:
            BxNx3 array, dropped batch of point clouds
    """
    batch_data = np.expand_dims(batch_data, axis=0)

    batch_, num_p, channel_ = batch_data.shape
    new_batch_data = []
    for batch_index in range(batch_):
        range_xyz = np.max(batch_data[batch_index, :, 0:2], axis=0) - np.min(batch_data[batch_index, :, 0:2], axis=0)

        crop_range = 0.5 + (np.random.rand(2) * 0.5)

        loop_count = 0
        while not check_aspect_2d(crop_range, 0.75):
            loop_count += 1
            crop_range = 0.5 + (np.random.rand(2) * 0.5)
            if loop_count > 100:
                break

        loop_count = 0

        while True:
            loop_count += 1
            new_range = range_xyz * crop_range  # / 2.0
            sample_center = batch_data[batch_index, np.random.choice(len(batch_data[batch_index])), 0:3]
            max_xyz = sample_center[:2] + new_range
            min_xyz = sample_center[:2] - new_range

            upper_idx = np.sum((batch_data[batch_index, :, :2] < max_xyz).astype(np.int32), 1) == 2
            lower_idx = np.sum((batch_data[batch_index, :, :2] > min_xyz).astype(np.int32), 1) == 2

            new_point_idx = (upper_idx & lower_idx)

            # avoid having too small point clouds
            if (loop_count > 100) or (np.sum(new_point_idx) > 30000):
                break

        new_batch_data.append(batch_data[batch_index, new_point_idx, :])

    new_batch_data = np.array(new_batch_data)

    return np.squeeze(new_batch_data, axis=0)


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    batch_n, num_p, channel_n = batch_data.shape
    assert (clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(batch_n, num_p, channel_n), -1 * clip, clip)
    jittered_data += batch_data
    return jittered_data


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


class ScanNet(Dataset):
    def __init__(self, data_root='/home/lee/Documents/dataset/scannet/Ours/train', pre_training=True, test_=False):
        super().__init__()
        self.augmented_dir = "/home/lee/Documents/dataset/scannet/augmented_views"
        test_file_dir = '/home/lee/Documents/dataset/scannet/Ours/test'

        self.pre_training = pre_training
        self.dataset_path = []
        self.test_ = test_

        for i in os.listdir(data_root):
            self.dataset_path.append(os.path.join(data_root, i))

        if self.pre_training:
            for i in os.listdir(test_file_dir):
                self.dataset_path.append(os.path.join('/home/lee/Documents/dataset/scannet/Ours/test', i))

        if self.test_:
            self.dataset_path = []
            for i in os.listdir(test_file_dir):
                self.dataset_path.append(os.path.join(test_file_dir, i))

    def transforms(self, points):
        if self.pre_training:
            points = np.expand_dims(points, axis=0)
            points[:, :, :3] = rotate_point_cloud(points[:, :, :3])
            points[:, :, :3] = rotate_perturbation_point_cloud(points[:, :, :3])
            points[:, :, :3] = random_scale_point_cloud(points[:, :, :3])
            points[:, :, :3] = random_flip_point_cloud(points[:, :, :3])
            points[:, :, :3] = jitter_point_cloud(points[:, :, :3])
            points = random_drop_n_cuboids(points)

            return np.squeeze(points, axis=0)

    def _get_augmented_item(self, index):
        cluster_path = os.path.join(self.augmented_dir, 'station_{}.npy'.format(index + 1))
        if os.path.isfile(cluster_path):
            points_set = np.load(cluster_path)
        else:
            points_set = np.load(self.dataset_path[index])
            points_set = points_set[:, :6]
            points_set = cluster_pc(points_set, n_cluster=50)

            np.save(cluster_path, points_set)

        points_i = random_cuboid_point_cloud(points_set.copy())
        points_i = self.transforms(points_i)
        points_j = random_cuboid_point_cloud(points_set.copy())
        points_j = self.transforms(points_j)

        return points_i, points_j

    def _get_item(self, index):
        points_set = np.load(self.dataset_path[index])

        if self.test_:
            points = points_set[:, :6]
            seg_l = np.zeros((points.shape[0]))
        else:
            points = points_set[:, :6]
            seg_l = points_set[:, 6].astype(int)
        # seg_l = seg_l.reshape((-1))
        # seg_l = np.expand_dims(seg_l, axis=-1)

        return points, seg_l

    def __getitem__(self, item):
        if self.pre_training:
            return self._get_augmented_item(item)
        else:
            return self._get_item(item)

    def __len__(self):
        return len(self.dataset_path)


class ScanNetDatasetV2(ScanNet):
    def __init__(self, data_root="/home/lee/Documents/dataset/scannet/scannet_ptv3/train", pre_train=True, test_=False):
        super().__init__()

        self.val_file_dir = "/home/lee/Documents/dataset/scannet/scannet_ptv3/val"
        self.test_file_dir = "/home/lee/Documents/dataset/scannet/scannet_ptv3/test"

        self.pre_training = pre_train
        self.dataset_path = []
        self.test_ = test_

        for i in os.listdir(data_root):
            self.dataset_path.append(os.path.join(data_root, i))

        for i in os.listdir(self.val_file_dir):
            self.dataset_path.append(os.path.join(self.val_file_dir, i))

        if self.pre_training:
            for i in os.listdir(self.test_file_dir):
                self.dataset_path.append(os.path.join(self.test_file_dir, i))

        if self.test_:
            self.dataset_path = []
            for i in os.listdir(self.test_file_dir):
                self.dataset_path.append(os.path.join(self.test_file_dir, i))

    def _get_augmented_item(self, index):
        cluster_path = os.path.join(self.augmented_dir, 'station_{}.npy'.format(index + 1))
        if os.path.isfile(cluster_path):
            points_set = np.load(cluster_path)
        else:
            points_set = torch.load(self.dataset_path[index])
            points_set = np.hstack([points_set['coord']-np.mean(points_set['coord'], 0), points_set['color'] / 255])
            points_set = cluster_pc(points_set, n_cluster=50)

            np.save(cluster_path, points_set)

        points_i = random_cuboid_point_cloud(points_set.copy())
        points_i = self.transforms(points_i)
        points_j = random_cuboid_point_cloud(points_set.copy())
        points_j = self.transforms(points_j)

        return points_i, points_j

    def _get_item(self, index):
        points_set = torch.load(self.dataset_path[index])
        if self.test_:
            points = np.hstack([points_set['coord']-np.mean(points_set['coord'], 0), points_set['color'] / 255])
            seg_l = np.zeros((points.shape[0]))
        else:
            points = np.hstack([points_set['coord']-np.mean(points_set['coord'], 0), points_set['color'] / 255])
            seg_l = points_set['semantic_gt20']

        return points, seg_l

    def __getitem__(self, item):
        if self.pre_training:
            return self._get_augmented_item(item)
        else:
            return self._get_item(item)


if __name__ == '__main__':
    train_dataset_ = ScanNetDatasetV2(pre_train=False)
    train_dataloader_ = DataLoader(train_dataset_)
    for i_ in tqdm(train_dataloader_, total=len(train_dataloader_)):
        print(i_)
