from a_model import *
from dataset import *

from tqdm import tqdm
import matplotlib.pyplot as plt

import torch.utils.data


def cal_miou(predi, target):
    predi_choice = predi.data.max(1)[1]
    predi_np = predi_choice.cpu().data.numpy()
    target_np = target.cpu().data.numpy()
    # target_np = target_np.squeeze(-1)

    parts = range(13)
    part_iou = []
    for part in parts:
        iou_i = np.sum(np.logical_and(predi_np == part, target_np == part))
        iou_u = np.sum(np.logical_or(predi_np == part, target_np == part))
        if iou_u == 0:
            iou = 0
        else:
            iou = iou_i / float(iou_u)

        part_iou.append(iou)

    return part_iou, np.mean(part_iou)


def cal_per_part_iou(predi, target):
    predi_choice = predi.data.max(1)[1]
    predi_np = predi_choice.cpu().data.numpy()
    target_np = target.cpu().data.numpy()
    # target_np = target_np.squeeze(-1)

    predi_np = predi_np.reshape(16, -1)
    target_np = target_np.reshape(16, -1)

    parts = range(13)
    part_iou = []
    for part in parts:
        shape_iou = []
        for shape_idx in range(target_np.shape[0]):
            iou_i = np.sum(np.logical_and(predi_np[shape_idx] == part, target_np[shape_idx] == part))
            iou_u = np.sum(np.logical_or(predi_np[shape_idx] == part, target_np[shape_idx] == part))
            if iou_u == 0:
                iou = 0
            else:
                iou = iou_i / float(iou_u)

            shape_iou.append(iou)

        part_iou.append(np.mean(shape_iou))

    return part_iou


def main():
    device = torch.device("cuda:0")

    a = torch.load('model_result/model_s3dis.pth')
    b = torch.load('model_result/model_head_s3dis.pth')

    collect_fun = SparseCollation(voxel_size=0.07)

    test_dataset = S3DIS('test')
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=8, collate_fn=collect_fun, shuffle=True,
                                                  num_workers=8)

    model = OursModel(6, is_cat=True).to(device)
    model_head = SegmentationClassifierHead(32, 13).to(device)

    model.load_state_dict(a)
    model_head.load_state_dict(b)

    model.eval()
    model_head.eval()

    predi_list, shape_iou, part_iou = [], [], []
    for data in tqdm(test_dataloader, total=len(test_dataloader)):
        coord, feats, sem_label = data
        sem_label = sem_label.to(device).long()
        coord = ME.SparseTensor(feats.float(), coord, device=device)

        coord = model(coord)
        predi_ = model_head(coord)

        per_part_iou, iou = cal_miou(predi_, sem_label)
        # per_part_iou = cal_per_part_iou(predi_, sem_label)
        part_iou.append(per_part_iou)
        shape_iou.append(iou)

    part_iou = np.array(part_iou)

    print('shape_iou_Mean:', np.mean(shape_iou))
    print('per_shape_iou:', np.mean(part_iou, axis=0))


if __name__ == '__main__':
    main()
