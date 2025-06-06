import torch
import torch.nn.functional as F


def losses(b, a, T, margin, λ=0.7, mask=None, stop_gradient=False,teacher_weights=[0.42,1]):
    """
    b: List of teacher features
    a: List of student features
    mask: Binary mask, where 0 for normal and 1 for abnormal
    T: Temperature coefficient
    margin: Hyperparameter for controlling the boundary
    λ: Hyperparameter for balancing loss
    """

    if teacher_weights is None:
        teacher_weights = [1.0 / len(b)] * len(b)

    loss = 0.0
    margin_loss_n = 0.0
    margin_loss_a = 0.0
    contra_loss = 0.0

    for i in range(len(a)):
        s_ = a[i]
        t_ = b[i].detach() if stop_gradient else b[i]

        n, c, h, w = s_.shape

        s = s_.view(n, c, -1).transpose(1, 2)  # (N, H*W, C)
        t = t_.view(n, c, -1).transpose(1, 2)  # (N, H*W, C)

        s_norm = F.normalize(s, p=2, dim=2)
        t_norm = F.normalize(t, p=2, dim=2)

        cos_loss = 1 - F.cosine_similarity(s_norm, t_norm, dim=2)
        cos_loss = cos_loss.mean()*teacher_weights[0] if i<3 else cos_loss.mean()*teacher_weights[1]

        simi = torch.matmul(s_norm, t_norm.transpose(1, 2)) / T
        simi = torch.exp(simi)
        simi_sum = simi.sum(dim=2, keepdim=True)
        simi = simi / (simi_sum + 1e-8)
        diag_sim = torch.diagonal(simi, dim1=1, dim2=2)


        if mask is None:
            contra_loss = -torch.log(diag_sim + 1e-8).mean()
            margin_loss_n = F.relu(margin - diag_sim).mean()


        else:
            # gt label
            if len(mask.size()) < 3:
                normal_mask = (mask == 0)
                abnormal_mask = (mask == 1)
            # gt mask
            else:
                mask_ = F.interpolate(mask, size=(h, w), mode='nearest').squeeze(1)
                mask_flat = mask.view(mask_.size(0), -1)

                normal_mask = (mask_flat == 0)
                abnormal_mask = (mask_flat == 1)

            if normal_mask.sum() > 0:
                diag_sim_normal = diag_sim[normal_mask]
                contra_loss = -torch.log(diag_sim_normal + 1e-8).mean()
                margin_loss_n = F.relu(margin - diag_sim_normal).mean()
            if abnormal_mask.sum() > 0:
                diag_sim_abnormal = diag_sim[abnormal_mask]
                margin_loss_a = F.relu(diag_sim_abnormal - margin / 2).mean()

        margin_loss = teacher_weights[0]*margin_loss_n + teacher_weights[1]*margin_loss_a

        contra_weight = teacher_weights[0] if i < 3 else teacher_weights[1]
        loss += cos_loss * λ + contra_loss * contra_weight*(1-λ) + margin_loss

    return loss


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=15, stride=1, padding=7) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))                                            
    wiou = 1 - (inter + 1) / (union - inter + 1)
    #dice loss
    smooth = 1.0
    intersection = (pred * mask).sum(dim=(2, 3))
    dice = (2 * intersection + smooth) / (pred.sum(dim=(2, 3)) + mask.sum(dim=(2, 3)) + smooth)
    dice_loss = 1 - dice
    return (wbce + wiou + dice_loss).mean()

if __name__ == '__main__':
    a = torch.tensor([[0], [0], [0]])
    print(len(a.size()))

