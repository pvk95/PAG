"""
This file implements different loss functions. For the baseline models and the PAG:ct+pet models, only the composite
loss function (combination of dice coefficient and binary cross-entropy is important).
"""
import torch


class Loss_functions(object):
    def __init__(self, alpha_dice=1, alpha=1, beta=1, gamma=1, bce_loss=None):

        self.alpha_dice = alpha_dice
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        if bce_loss is not None:
            self.bce_loss = bce_loss
        else:
            self.bce_loss = torch.nn.BCELoss()

    @staticmethod
    def Dloss(seg_predict, seg_mask, smooth=1.0, eps=1e-7):

        # seg_predict = torch.sigmoid(seg_predict)
        intersection = torch.sum(seg_predict * seg_mask)
        den1 = torch.sum(seg_predict * seg_predict)
        den2 = torch.sum(seg_mask * seg_mask)
        dice = (2 * intersection + smooth) / (den1 + den2 + smooth + eps)

        return 1 - dice

    @staticmethod
    def PetReconLoss(pet_predict, pet, epsilon=1e-10):

        eps = pet * torch.abs(torch.clamp(pet_predict, min=0.01, max=0.98) - pet)
        factor = (1 - eps) / (eps + epsilon)

        beta = 0.5 * torch.log(factor)
        loss = torch.mean(torch.exp(-beta))

        return loss

    @staticmethod
    def AbsLoss(pet_predict, pet, epsilon=1e-10):
        loss = torch.mean(pet * torch.abs(pet_predict - pet))

        return loss

    def loss_reg(self, outputs, inputs):
        img_gt = outputs[0]
        img_recon = inputs[0]

        recon_loss = torch.mean(torch.pow((img_recon - img_gt), 2))

        return self.beta * recon_loss

    def loss_ct(self, outputs, inputs):
        seg_predict = outputs[2]
        seg_mask = inputs[2]

        dice_loss = self.Dloss(seg_predict=seg_predict, seg_mask=seg_mask)
        bce_loss = self.bce_loss(seg_predict, seg_mask)

        ind_loss = {'Dice': dice_loss.item(), 'CCE': bce_loss.item()}

        return self.alpha_dice * dice_loss + self.alpha * bce_loss, ind_loss

    def loss_ct_reg(self, outputs, inputs):
        img_recon = outputs[0]
        seg_predict = outputs[2]
        img_gt = inputs[0]
        seg_mask = inputs[2]

        smooth = 1.0

        intersection = torch.sum(seg_predict * seg_mask)
        den1 = torch.sum(seg_predict * seg_predict)
        den2 = torch.sum(seg_mask * seg_mask)
        dice_loss = -(2 * intersection + smooth) / (den1 + den2 + smooth + 1e-5) + 1

        bce_loss = self.bce_loss(seg_predict, seg_mask)

        recon_loss = torch.mean(torch.pow((img_recon - img_gt), 2))

        return dice_loss + self.alpha * bce_loss + self.beta * recon_loss

    def loss_ct_pet(self, outputs, inputs):
        pet_recon = outputs[1]
        seg_predict = outputs[2]
        pet_gt = inputs[1]
        seg_mask = inputs[2]

        dice_loss = self.Dloss(seg_predict=seg_predict, seg_mask=seg_mask)
        bce_loss = self.bce_loss(seg_predict, seg_mask)

        recon_loss = torch.mean(pet_gt * torch.pow((pet_recon - pet_gt), 2))
        # recon_loss = self.PetReconLoss(pet_predict=pet_recon, pet=pet_gt)
        # recon_loss = self.AbsLoss(pet_predict=pet_recon, pet=pet_gt)

        ind_loss = {'Dice': dice_loss.item(), 'CCE': bce_loss.item(), 'PET_Recon': recon_loss.item()}

        return self.alpha_dice * dice_loss + self.alpha * bce_loss + self.beta * recon_loss, ind_loss

    def loss_mask(self, outputs, inputs):
        pet_recon = outputs[1]
        seg_predict = outputs[2]
        seg_pet_predict = outputs[3]

        pet_gt = inputs[1]
        seg_mask = inputs[2]
        seg_pet = pet_gt * seg_mask

        dice_loss = self.Dloss(seg_predict=seg_predict, seg_mask=seg_mask)

        bce_loss = self.bce_loss(seg_predict, seg_mask)

        recon_loss = torch.mean(pet_gt * torch.pow((pet_recon - pet_gt), 2))

        # exp1-2
        seg_pet_loss = torch.mean(torch.pow((seg_pet_predict - seg_pet), 2))

        ind_loss = {'Dice': dice_loss.item(), 'CCE': bce_loss.item(), 'PET_Recon': recon_loss.item(),
                    'Seg_Pet': seg_pet_loss.item()}

        return self.alpha_dice * dice_loss + self.alpha * bce_loss + \
               self.beta * recon_loss + self.gamma * seg_pet_loss, ind_loss

    def loss_pet(self, outputs, inputs):
        pet_recon = outputs[1]
        pet_gt = inputs[1]

        seg_mask = inputs[2]
        err = torch.pow((pet_recon - pet_gt), 2)

        # cv0
        # recon_loss = torch.mean(1e4 * seg_mask * err + (1 - seg_mask) * err)

        # exp2
        recon_loss = torch.mean(1e3 * seg_mask * err + (1 - seg_mask) * err)

        ind_loss = {'Dice': 0, 'CCE': 0, 'PET_Recon': recon_loss.item()}

        # recon_loss = torch.mean(pet_gt * torch.pow((pet_recon - pet_gt), 2))
        # recon_loss = torch.mean(torch.abs(pet_recon - pet_gt))

        return self.beta * recon_loss, ind_loss
