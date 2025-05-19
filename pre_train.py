from model import OursModel, ProjectionHead
from dataset import ScanNet, SparseAugmentedCollation, ScanNetDatasetV2
from moco import MoCo

import torch
import torch.nn as nn
import torch.utils.data

import time
import numpy as np
from tqdm import tqdm
from functools import partial
import matplotlib.pyplot as plt
import MinkowskiEngine as ME

t = time.localtime()
t = (str(t.tm_year) + '_' + str(t.tm_mon) + '_' + str(t.tm_mday) + '_' +
     str(t.tm_hour) + '_' + str(t.tm_min) + '_' + str(t.tm_sec))


def array_to_sequence(batch_data):
    return [row for row in batch_data]


def array_to_torch_sequence(batch_data):
    return [torch.from_numpy(row).float() for row in batch_data]


def numpy_to_sparse_tensor(p_coord, p_feats, p_label=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p_coord = ME.utils.batched_coordinates(array_to_sequence(p_coord), dtype=torch.float32)
    p_feats = ME.utils.batched_coordinates(array_to_torch_sequence(p_feats), dtype=torch.float32)[:, 1:]

    if p_label is not None:
        p_label = ME.utils.batched_coordinates(array_to_torch_sequence(p_label), dtype=torch.float32)[:, 1:]

        return ME.SparseTensor(features=p_feats, coordinates=p_coord.int(), device=device), p_label.cuda()

    return ME.SparseTensor(features=p_feats, coordinates=p_coord.int(), device=device)


def collate_points_to_sparse_tensor(pi_coord, pi_feats, pj_coord, pj_feats):
    # voxelize on a sparse tensor
    points_i = numpy_to_sparse_tensor(pi_coord, pi_feats)
    points_j = numpy_to_sparse_tensor(pj_coord, pj_feats)

    return points_i, points_j


def cosine_schedule_with_warmup(k, num_epochs, batch_size, dataset_size):
    iter_per_epoch = (dataset_size + batch_size - 1) // batch_size

    return 0.5 * (1 + np.cos(np.pi * k / (num_epochs * iter_per_epoch)))


def main():
    collate_fn = SparseAugmentedCollation(resolution=0.07)
    train_dataset = ScanNetDatasetV2()
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, collate_fn=collate_fn,
                                               shuffle=True, num_workers=10)

    device = torch.device("cuda:0")
    criterion = nn.CrossEntropyLoss()

    model_ = MoCo(OursModel, ProjectionHead, d_in=6, in_channel=32, out_channel=128, is_cat=True).to(device)
    model_.train()

    optimizer = torch.optim.Adam(model_.parameters(), lr=1e-1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=partial(
            cosine_schedule_with_warmup, num_epochs=500,
            batch_size=8, dataset_size=len(train_loader) * 8
        )
    )

    loss_list = []
    for e_i in (range(500)):
        loss_sum = 0
        print("=========================== {} ===========================".format(e_i))

        for data in tqdm(train_loader, total=len(train_loader)):
            (xi_coord, xi_feats, si), (xj_coord, xj_feats, sj) = data
            xi, xj = collate_points_to_sparse_tensor(xi_coord, xi_feats, xj_coord, xj_feats)

            optimizer.zero_grad()

            out_seg, tgt_seg = model_(xi, xj, [si, sj])
            loss = criterion(out_seg, tgt_seg)
            loss.backward()

            loss_sum += loss.detach().cpu()

            optimizer.step()

        scheduler.step()
        loss_list.append(loss_sum / len(train_loader))

        print("loss: {}".format(loss_sum / len(train_loader)))
        print("learning rate: {}".format(scheduler.get_last_lr()[0]))

    torch.save(model_.model_q.state_dict(), 'model_result/pre_train_model_{}.pth'.format(t))

    plt.plot(loss_list, 'r', label='LOSS_CURVE')
    plt.legend()

    plt.savefig('model_result/pre_train_loss_{}.png'.format(t))

    plt.show()


if __name__ == '__main__':
    main()
