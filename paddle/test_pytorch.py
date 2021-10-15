import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RegWeightedL1Loss(nn.Module):
    def __init__(self):
        super(RegWeightedL1Loss, self).__init__()

    def forward(self, output, mask, ind, target, dep):

        dep=dep.squeeze(2)
        dep[dep<5]=dep[dep<5]*0.01
        print('dep.shape = ', dep.shape)

        dep[dep >= 5] = torch.log10(dep[dep >=5]-4)+0.1
        pred = transpose_and_gather_feat(output, ind)
        mask = mask.float()
        # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        #losss=torch.abs(pred * mask-target * mask)
        #loss = F.l1_loss(pred * mask, target * mask, size_average=False)
        loss=torch.abs(pred * mask-target * mask)
        loss=torch.sum(loss,dim=2)*dep
        loss=loss.sum()
        loss = loss / (mask.sum() + 1e-4)

        return loss
class RegWeightedL1Loss2(nn.Module):
    def __init__(self):
        super(RegWeightedL1Loss2, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = transpose_and_gather_feat(output, ind)
        mask = mask.float()
        # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        # print(pred.shape)
        # print(target)
        loss = F.l1_loss(pred * mask, target * mask, size_average=False)
        loss = loss / (mask.sum() + 1e-4)
        return loss
def gather_feat(feat, ind, mask=None):

    dim = feat.size(2)
    ind = ind.unsqueeze(2)
    print(ind.shape)
    ind = ind.expand(ind.size(0), ind.size(1), dim)  # batch * channel * num_obj
    print(ind.shape)
    print(ind[0][0])
    print(ind[1][0])
    print(ind.shape)
    print(feat.shape)
    feat = feat.gather(1, ind)
    print(feat.shape)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def transpose_and_gather_feat(feat, ind):
    '''

    Args:
        feat: feature from network, batch * channel * h * w
        ind: induce location for object in heatmap, batch * max_num_objects
    Returns:

    '''
    feat = feat.permute(0, 2, 3, 1).contiguous()  # batch * h * w * channel
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = gather_feat(feat, ind)
    return feat  # batch * (max_num_objects) * channel

def compute_rot_loss(output, target_bin, target_res, mask):
    # output: (B, 128, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos,
    #                 bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
    # target_bin: (B, 128, 2) [bin1_cls, bin2_cls]
    # target_res: (B, 128, 2) [bin1_res, bin2_res]
    # mask: (B, 128, 1)
    # import pdb; pdb.set_trace()
    output = output.view(-1, 8)
    target_bin = target_bin.view(-1, 2)
    target_res = target_res.view(-1, 2)
    mask = mask.view(-1, 1)
    loss_bin1 = compute_bin_loss(output[:, 0:2], target_bin[:, 0], mask)
    loss_bin2 = compute_bin_loss(output[:, 4:6], target_bin[:, 1], mask)
    loss_res = torch.zeros_like(loss_bin1)
    if target_bin[:, 0].nonzero().shape[0] > 0:
        idx1 = target_bin[:, 0].nonzero()[:, 0]
        valid_output1 = torch.index_select(output, 0, idx1.long())
        valid_target_res1 = torch.index_select(target_res, 0, idx1.long())
        loss_sin1 = compute_res_loss(
            valid_output1[:, 2], torch.sin(valid_target_res1[:, 0]))
        loss_cos1 = compute_res_loss(
            valid_output1[:, 3], torch.cos(valid_target_res1[:, 0]))
        loss_res += loss_sin1 + loss_cos1
    if target_bin[:, 1].nonzero().shape[0] > 0:
        idx2 = target_bin[:, 1].nonzero()[:, 0]
        valid_output2 = torch.index_select(output, 0, idx2.long())
        valid_target_res2 = torch.index_select(target_res, 0, idx2.long())
        loss_sin2 = compute_res_loss(
            valid_output2[:, 6], torch.sin(valid_target_res2[:, 1]))
        loss_cos2 = compute_res_loss(
            valid_output2[:, 7], torch.cos(valid_target_res2[:, 1]))
        loss_res += loss_sin2 + loss_cos2
    return loss_bin1 + loss_bin2 + loss_res

def compute_bin_loss(output, target, mask):
    print(mask)
    print(output.shape)
    print(mask.shape)
    mask = mask.expand_as(output)
    print(mask)
    print(mask.shape)

    output = output * mask.float()
    return F.cross_entropy(output, target, reduction='elementwise_mean')

class BinRotLoss(nn.Module):
    def __init__(self):
        super(BinRotLoss, self).__init__()

    def forward(self, output, mask, ind, rotbin, rotres):
        pred = transpose_and_gather_feat(output, ind)
        loss = compute_rot_loss(pred, rotbin, rotres, mask)
        return loss


def compute_res_loss(output, target):
    return F.smooth_l1_loss(output, target, reduction='elementwise_mean')

def check():
    num_joints = 25
    num_object = 64
    batch = 2
    channel = 8
    width = 100
    height = 50
    feat = torch.arange(batch * channel * width * height, dtype=torch.float32).reshape(batch, channel, height, width)
    induce = torch.zeros((2, 64), dtype=torch.int64)  # batch, num_obj
    batch_hp = torch.arange(batch * num_object * channel, dtype=torch.float32).reshape((batch, num_object, channel))
    hps_mask = torch.ones((batch, num_object, channel), dtype=torch.float32)
    dep = torch.arange((batch * num_object * 1), dtype=torch.float32).reshape((batch, num_object, 1))
    induce[0][0] = 55
    induce[1][0] = 185
    # p2d_l1loss = RegWeightedL1Loss()
    # lossp2d = p2d_l1loss(feat, hps_mask, induce, batch_hp, dep)
    #
    # p3d_l1loss = RegWeightedL1Loss2()
    # lossp3d = p3d_l1loss(feat, hps_mask, induce, batch_hp)
    # print(lossp3d)
    # rot_mask = torch.ones((batch, num_object), dtype=torch.float32)
    # rot_bin = torch.ones((batch, num_object, 2), dtype=torch.int64)
    # rot_res = torch.ones((batch, num_object, 2), dtype=torch.float32)
    # crit_rot = BinRotLoss()
    # rot_loss = crit_rot(feat, rot_mask, induce, rot_bin, rot_res)
    # print(rot_loss)

    cys = (induce / width).int().float()
    cxs = (induce % width).int().float()
    cxs = cxs.view(batch, num_object, 1).expand(batch, num_object, num_joints)
    print(cxs)

    rot = torch.arange(batch * num_object * 8, dtype=torch.float32).reshape(batch, num_object, 8)

    kps = torch.zeros((batch, num_object, 2 * num_joints))
    si = torch.zeros_like(kps[:, :, 0:1]) + 721
    alpha_idx = rot[:, :, 1] > rot[:, :, 5]
    alpha_idx = alpha_idx.float()
    alpha1 = torch.atan(rot[:, :, 2] / rot[:, :, 3]) + (-0.5 * np.pi)
    alpha2 = torch.atan(rot[:, :, 6] / rot[:, :, 7]) + (0.5 * np.pi)
    alpna_pre = alpha1 * alpha_idx + alpha2 * (1 - alpha_idx)
    alpna_pre = alpna_pre.unsqueeze(2)
    # print(alpna_pre)


if __name__ == '__main__':
    pinv = np.load('/home/beta/baidu/personal-code/autoshape/kitti_data/test/pinv.npy')
    dim = np.load('/home/beta/baidu/personal-code/autoshape/kitti_data/test/dim.npy')
    kps = np.load('/home/beta/baidu/personal-code/autoshape/kitti_data/test/kps.npy')
    p3d = np.load('/home/beta/baidu/personal-code/autoshape/kitti_data/test/p3d.npy')
    rot = np.load('/home/beta/baidu/personal-code/autoshape/kitti_data/test/rot.npy')

    print(pinv.shape)