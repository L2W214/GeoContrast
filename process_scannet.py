import os
import open3d as o3d
from plyfile import PlyData
import numpy as np
from tqdm import tqdm


def watch_array(array_):
    pcd_def = o3d.geometry.PointCloud()
    pcd_def.points = o3d.utility.Vector3dVector(array_[:, :3])
    pcd_def.colors = o3d.utility.Vector3dVector(array_[:, 3:6])

    o3d.visualization.draw_geometries([pcd_def])


def read_ply(file):
    with open(file, 'rb') as f:
        ply_data = PlyData.read(f)
        num_vert = ply_data['vertex'].count
        vertices = np.zeros(shape=[num_vert, 6], dtype=np.float32)
        vertices[:, 0] = ply_data['vertex'].data['x']
        vertices[:, 1] = ply_data['vertex'].data['y']
        vertices[:, 2] = ply_data['vertex'].data['z']
        vertices[:, 3] = ply_data['vertex'].data['red']
        vertices[:, 4] = ply_data['vertex'].data['green']
        vertices[:, 5] = ply_data['vertex'].data['blue']

    return vertices


def read_label(file):
    with open(file, 'rb') as f:
        ply_data = PlyData.read(f)
        num_vert = ply_data['vertex'].count
        label_def = np.zeros(shape=[num_vert, 1], dtype=np.int64)
        label_def[:, 0] = ply_data['vertex'].data['label']

    return label_def


DATA_PATH = '/home/lee/Documents/dataset/scannet/RawData'
TRAIN_FOLDER = os.listdir(os.path.join(DATA_PATH, 'scans'))
TEST_FOLDER = os.listdir(os.path.join(DATA_PATH, 'scans_test'))

# for i in tqdm(TRAIN_FOLDER, total=len(TRAIN_FOLDER)):
#     out_filename = os.path.join('/home/lee/Documents/dataset/scannet/Ours/train', i+'.npy')
#     sub_folder = os.path.join(DATA_PATH, 'scans', i)
#     sub_content = os.listdir(sub_folder)
#     for j in sub_content:
#         if '.ply' in j and 'labels' in j:
#             label_ = read_label(os.path.join(DATA_PATH, 'scans', i, j))
#
#         if '.ply' in j and 'labels' not in j:
#             point_cloud_ = read_ply(os.path.join(DATA_PATH, 'scans', i, j))
#             point_cloud_[:, 3:6] = point_cloud_[:, 3:6] / 255
#             # watch_array(point_cloud_)
#
#     pc_l = np.concatenate([point_cloud_, label_], axis=1)
#     np.save(out_filename, pc_l)

for i in tqdm(TEST_FOLDER, total=len(TEST_FOLDER)):
    out_filename = os.path.join('/home/lee/Documents/dataset/scannet/Ours/test', i+'.npy')
    sub_folder = os.path.join(DATA_PATH, 'scans_test', i)
    sub_content = os.listdir(sub_folder)
    for j in sub_content:
        point_cloud_ = read_ply(os.path.join(DATA_PATH, 'scans_test', i, j))
        point_cloud_[:, 3:6] = point_cloud_[:, 3:6] / 255
        # watch_array(point_cloud_)

        np.save(out_filename, point_cloud_)
