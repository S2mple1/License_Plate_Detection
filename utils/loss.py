import torch
import torch.nn as nn
import numpy as np
from general import bbox_iou
from torch_utils import is_parallel


def smooth_BCE(eps=0.1):
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    def __init__(self, alpha=0.05):
        super(BCEBlurWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)
        dx = pred - true
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss = loss * alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred_prob = torch.sigmoid(pred)
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss = loss * alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class QFocalLoss(nn.Module):
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(QFocalLoss, self).__init__()
        self.loss_fcn = loss_fcn
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred_prob = torch.sigmoid(pred)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss = loss * alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class WingLoss(nn.Module):
    def __init__(self, w=10, e=2):
        super(WingLoss, self).__init__()
        self.w = w
        self.e = e
        self.C = self.w - self.w * np.log(1 + self.w / self.e)

    def forward(self, x, t, sigma=1):
        weight = torch.ones_like(t)
        weight[torch.where(t == -1)] = 0
        diff = weight * (x - t)
        abs_diff = diff.abs()
        flag = (abs_diff.data < self.w).float()
        y = flag * self.w * torch.log(1 + abs_diff / self.e) + (1 - flag) * (abs_diff - self.C)
        return y.sum()


class LandmarksLoss(nn.Module):
    def __init__(self, alpha=1.0):
        super(LandmarksLoss, self).__init__()
        self.loss_fcn = WingLoss()
        self.alpha = alpha

    def forward(self, pred, truel, mask):
        loss = self.loss_fcn(pred * mask, truel * mask)
        return loss / (torch.sum(mask) + 10e-14)


def compute_loss(p, targets, model):
    device = targets.device
    lcls, lbox, lobj, lmark = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1,
                                                                                                        device=device), torch.zeros(
        1, device=device)
    tcls, tbox, indices, anchors, tlandmarks, lmks_mask = build_targets(p, targets, model)
    h = model.hyp

    BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
    BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

    landmarks_loss = LandmarksLoss(1.0)

    cp, cn = smooth_BCE(eps=0.0)

    # Focal loss
    g = h['fl_gamma']
    if g > 0:
        BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

    # Losses
    nt = 0
    no = len(p)
    balance = [4.0, 1.0, 0.4] if no == 3 else [4.0, 1.0, 0.4, 0.1]
    for i, pi in enumerate(p):
        b, a, gj, gi = indices[i]
        tobj = torch.zeros_like(pi[..., 0], device=device)

        n = b.shape[0]
        if n:
            nt = nt + n
            ps = pi[b, a, gj, gi]

            # Regression
            pxy = ps[:, :2].sigmoid() * 2. - 0.5
            pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
            pbox = torch.cat((pxy, pwh), 1)  # predicted box
            iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)
            lbox = lbox + (1.0 - iou).mean()  # iou loss

            # Objectness
            tobj[b, a, gj, gi] = (1.0 - model.gr) + model.gr * iou.detach().clamp(0).type(tobj.dtype)

            # Classification
            if model.nc > 1:
                t = torch.full_like(ps[:, 13:], cn, device=device)
                t[range(n), tcls[i]] = cp
                lcls = lcls + BCEcls(ps[:, 13:], t)  # BCE

            plandmarks = ps[:, 5:13]
            plandmarks_02 = plandmarks[:, 0:2] * anchors[i]
            plandmarks_24 = plandmarks[:, 2:4] * anchors[i]
            plandmarks_46 = plandmarks[:, 4:6] * anchors[i]
            plandmarks_68 = plandmarks[:, 6:8] * anchors[i]
            plandmarks_8 = plandmarks[:, 8:]
            plandmarks = torch.cat((plandmarks_02, plandmarks_24, plandmarks_46, plandmarks_68, plandmarks_8), dim=-1)

            lmark = lmark + landmarks_loss(plandmarks, tlandmarks[i], lmks_mask[i])

        lobj = lobj + BCEobj(pi[..., 4], tobj) * balance[i]

    s = 3 / no
    lbox = lbox * h['box'] * s
    lobj = lobj * h['obj'] * s * (1.4 if no == 4 else 1.)
    lcls = lcls * h['cls'] * s
    lmark = lmark * h['landmark'] * s

    bs = tobj.shape[0]

    loss = lbox + lobj + lcls + lmark
    return loss * bs, torch.cat((lbox, lobj, lcls, lmark, loss)).detach()


def build_targets(p, targets, model):
    det = model.module.model[-1] if is_parallel(model) else model.model[-1]
    na, nt = det.na, targets.shape[0]
    tcls, tbox, indices, anch, landmarks, lmks_mask = [], [], [], [], [], []
    gain = torch.ones(15, device=targets.device)
    ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)
    targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)

    g = 0.5  # bias
    off = torch.tensor([[0, 0],
                        [1, 0], [0, 1], [-1, 0], [0, -1],
                        ], device=targets.device).float() * g

    for i in range(det.nl):
        anchors, shape = det.anchors[i], p[i].shape
        gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]
        gain[6:14] = torch.tensor(p[i].shape)[[3, 2, 3, 2, 3, 2, 3, 2]]

        t = targets * gain
        if nt:
            r = t[:, :, 4:6] / anchors[:, None]
            j = torch.max(r, 1. / r).max(2)[0] < model.hyp['anchor_t']
            t = t[j]

            # Offsets
            gxy = t[:, 2:4]
            gxi = gain[[2, 3]] - gxy
            j, k = ((gxy % 1. < g) & (gxy > 1.)).T
            l, m = ((gxi % 1. < g) & (gxi > 1.)).T
            j = torch.stack((torch.ones_like(j), j, k, l, m))
            t = t.repeat((5, 1, 1))[j]
            offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
        else:
            t = targets[0]
            offsets = 0

        # Define
        b, c = t[:, :2].long().T
        gxy = t[:, 2:4]
        gwh = t[:, 4:6]
        gij = (gxy - offsets).long()
        gi, gj = gij.T

        # Append
        a = t[:, 14].long()
        indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))
        tbox.append(torch.cat((gxy - gij, gwh), 1))
        anch.append(anchors[a])
        tcls.append(c)

        # landmarks
        lks = t[:, 6:14]
        lks_mask = torch.where(lks < 0, torch.full_like(lks, 0.), torch.full_like(lks, 1.0))

        lks[:, [0, 1]] = (lks[:, [0, 1]] - gij)
        lks[:, [2, 3]] = (lks[:, [2, 3]] - gij)
        lks[:, [4, 5]] = (lks[:, [4, 5]] - gij)
        lks[:, [6, 7]] = (lks[:, [6, 7]] - gij)

        lks_mask_new = lks_mask
        lmks_mask.append(lks_mask_new)
        landmarks.append(lks)

    return tcls, tbox, indices, anch, landmarks, lmks_mask
