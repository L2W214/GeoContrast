from a_model import *
from dataset import *

from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.utils.data


def main():
    device = torch.device("cuda:0")

    collect_fun = SparseCollation(voxel_size=0.07)

    # train_dataset = S3DIS()
    # test_dataset = S3DIS('test')
    train_dataset = ScanNetDatasetV2(pre_train=False)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=16, collate_fn=collect_fun, shuffle=True,
                                                   num_workers=20)
    # test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=8, collate_fn=collect_fun, shuffle=True,
    #                                               num_workers=8)

    pre_parameter = torch.load("model_result/pre_train_model_2025_5_7_19_55_58.pth")
    model = OursModel(6, is_cat=True).to(device)
    model.load_state_dict(pre_parameter)
    model_head = SegmentationClassifierHead(32, 50).to(device)

    model.train()
    model_head.train()

    criterion = torch.nn.CrossEntropyLoss()
    optim_params = list(model.parameters()) + list(model_head.parameters())

    optimizer = torch.optim.SGD(optim_params, lr=2.4e-2, momentum=0.9, weight_decay=1.0e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)

    acc_list, loss_list = [], []
    for e_i in range(100):
        loss_sum = 0
        accuracy_sum = 0
        print("=========================== {} ===========================".format(e_i))
        for data in tqdm(train_dataloader, total=len(train_dataloader)):
            coord, feats, sem_label = data
            sem_label = sem_label.to(device).long()
            coord = ME.SparseTensor(feats.float(), coord, device=device)

            optimizer.zero_grad()

            coord = model(coord)
            predict_label = model_head(coord)

            loss = criterion(predict_label, sem_label.long())
            loss.backward()

            optimizer.step()

            predict_choice = predict_label.data.max(dim=1)[1]
            correct = predict_choice.eq(sem_label.data).cpu().sum()

            loss_sum += loss.detach().cpu()
            accuracy_sum += correct.item() / float(sem_label.shape[0])

        scheduler.step()

        loss_mean = loss_sum / len(train_dataloader)
        loss_list.append(loss_mean)

        acc_mean = accuracy_sum / len(train_dataloader)
        acc_list.append(acc_mean)

        print("loss: {}".format(loss_mean))

    torch.save(model.state_dict(), 'model_result/model_scannet.pth')
    torch.save(model_head.state_dict(), 'model_result/model_head_scannet.pth')

    plt.plot(loss_list, 'r', label='LOSS_CURVE')
    plt.plot(acc_list, 'b', label='ACCURACY_CURVE')
    plt.title("accuracy: {}".format(acc_list[-1]))
    plt.legend()

    plt.show()


if __name__ == '__main__':
    main()
