import torch
from fastai.losses import BaseLoss

class DiceLoss(BaseLoss):
    def __init__(self, axis=1, smooth=1e-8, loss_cls=None):
        super().__init__(axis=axis)
        self.smooth = smooth

    def forward(self, pred, targ):
        pred = pred.argmax(dim=self.axis)
        pred = torch.softmax(pred, dim=self.axis)  # Softmax untuk mendapatkan probabilitas kelas
        targ = targ.float()  # Mengubah target menjadi float
        intersection = torch.sum(pred * targ, dim=self.axis)  # Menghitung intersection antara prediksi dan target
        union = torch.sum(pred + targ, dim=self.axis)  # Menghitung union antara prediksi dan target
        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)  # Menghitung skor Dice
        return 1 - dice_score.mean()  # Mengembalikan loss (1 - skor Dice)

loss_func = DiceLoss(loss_cls=torch.nn.BCELoss())
