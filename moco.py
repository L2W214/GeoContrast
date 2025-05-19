import torch.nn as nn
import torch

import numpy as np
import MinkowskiEngine as ME


def list_segments_points(p_coord, p_feats, labels):
    c_coord = []
    c_feats = []

    seg_batch_count = 0

    for batch_num in range(labels.shape[0]):
        for segment_lbl in np.unique(labels[batch_num]):
            if segment_lbl == -1:
                continue

            batch_ind = p_coord[:, 0] == batch_num
            segment_ind = labels[batch_num] == segment_lbl

            # we are listing from sparse tensor, the first column is the batch index, which we drop
            segment_coord = p_coord[batch_ind][segment_ind][:, :]
            segment_coord[:, 0] = seg_batch_count
            seg_batch_count += 1

            segment_feats = p_feats[batch_ind][segment_ind]

            c_coord.append(segment_coord)
            c_feats.append(segment_feats)

    seg_coord = torch.vstack(c_coord)
    seg_feats = torch.vstack(c_feats)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return ME.SparseTensor(features=seg_feats, coordinates=seg_coord, device=device)


class MoCo(nn.Module):
    def __init__(self, model, model_head, d_in, in_channel, out_channel, is_cat, k=65536, m=0.999, t=0.1):
        super().__init__()

        self.k = k
        self.m = m
        self.t = t

        self.model_q = model(in_channels=d_in, is_cat=is_cat)
        self.head_q = model_head(in_channels=in_channel, out_channels=out_channel)

        self.model_k = model(in_channels=d_in, is_cat=is_cat)
        self.head_k = model_head(in_channels=in_channel, out_channels=out_channel)

        for param_q, param_k in zip(self.model_q.parameters(), self.model_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        for param_q, param_k in zip(self.head_q.parameters(), self.head_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        self.register_buffer('queue_pcd', torch.randn(128, k))
        self.queue_pcd = nn.functional.normalize(self.queue_pcd, dim=0)

        self.register_buffer('queue_seg', torch.randn(128, k))
        self.queue_seg = nn.functional.normalize(self.queue_seg, dim=0)

        self.register_buffer('queue_pcd_ptr', torch.zeros(1, dtype=torch.long))
        self.register_buffer('queue_seg_ptr', torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.model_q.parameters(), self.model_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1 - self.m)

        for param_q, param_k in zip(self.head_q.parameters(), self.head_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue_pcd(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_pcd_ptr)
        if ptr + batch_size <= self.k:
            self.queue_pcd[:, ptr: ptr + batch_size] = keys.T
        else:
            tail_size = self.k - ptr
            head_size = batch_size - tail_size
            self.queue_pcd[:, ptr: self.k] = keys.T[:, :tail_size]
            self.queue_pcd[:, :head_size] = keys.T[:, tail_size:]

        ptr = (ptr + batch_size) % self.k
        self.queue_pcd_ptr[0] = ptr

    @torch.no_grad()
    def _dequeue_and_enqueue_seg(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_seg_ptr)
        # ============= 在非稀疏张量应用过程中，可能会存在一些问题
        if ptr + batch_size <= self.k:
            self.queue_seg[:, ptr: ptr + batch_size] = keys.T
        else:
            tail_size = self.k - ptr
            head_size = batch_size - tail_size
            self.queue_seg[:, ptr: self.k] = keys.T[:, :tail_size]
            self.queue_seg[:, :head_size] = keys.T[:, tail_size:]

        ptr = (ptr + batch_size) % self.k
        self.queue_seg_ptr[0] = ptr

    def forward(self, pcd_q, pcd_k, segments=None):
        h_q = self.model_q(pcd_q)
        if segments is None:
            z_q = self.head_q(h_q)
            q_pcd = nn.functional.normalize(z_q, dim=1)
        else:
            h_qs = list_segments_points(h_q.C, h_q.F, segments[0])
            z_qs = self.head_q(h_qs)
            q_seg = nn.functional.normalize(z_qs, dim=1)

        with torch.no_grad():
            self._momentum_update_key_encoder()

            h_k = self.model_k(pcd_k)
            if segments is None:
                z_k = self.head_k(h_k)
                k_pcd = nn.functional.normalize(z_k, dim=1)
            else:
                h_ks = list_segments_points(h_k.C, h_k.F, segments[1])
                z_ks = self.head_k(h_ks)
                k_seg = nn.functional.normalize(z_ks, dim=1)

        if segments is None:
            q_pcd, k_pcd = q_pcd.squeeze(), k_pcd.squeeze()

        if segments is None:
            l_pos_pcd = torch.einsum('nc, nc->n', [q_pcd, k_pcd]).unsqueeze(-1)
            l_neg_pcd = torch.einsum('nc, ck->nk', [q_pcd, self.queue_pcd.clone().detach()])

            logits_pcd = torch.cat([l_pos_pcd, l_neg_pcd], dim=1)
            logits_pcd /= self.t
            labels_pcd = torch.zeros(logits_pcd.shape[0], dtype=torch.long).cuda()

            self._dequeue_and_enqueue_pcd(k_pcd)

            return logits_pcd, labels_pcd
        else:
            l_pos_seg = torch.einsum('nc, nc->n', [q_seg, k_seg]).unsqueeze(-1)
            l_neg_seg = torch.einsum('nc, ck->nk', [q_seg, self.queue_seg.clone().detach()])

            logits_seg = torch.cat([l_pos_seg, l_neg_seg], dim=1)
            logits_seg /= self.t
            labels_seg = torch.zeros(logits_seg.shape[0], dtype=torch.long).cuda()

            self._dequeue_and_enqueue_pcd(k_seg)

            return logits_seg, labels_seg
