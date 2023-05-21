from DataSet import DRIVE
from torchvision import transforms
from torch.autograd import Variable
from matplotlib import pyplot as plt
import time
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score
from models import *
import numpy as np
# from diceloss import *
from LossFunctions import *

from fastai.vision.all import *
from DatasetHandler_Augmentation import *
from fastai.text.all import *
from fastai.collab import *
from fastai.tabular.all import *

dataset = DRIVE("/content/drive/MyDrive/tinyunet/DRIVE",(512,512))

filters = [8,16,32,64]
#model = TinyUNet_AAFx1(3,1,filters)
model = TinyUNet_AAFx14(3,1,filters)
model.cuda()

#import torchsummary
#torchsummary.summary(model,(3,512,512))

normalizer = transforms.Normalize(mean=dataset.Train.get_mean(),std=dataset.Train.get_std())

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, min_lr=0.0000001, patience=100, verbose=True)
loss_function = DICELoss()
# loss_function = nn.HuberLoss()

model.train()
max_iter = 20000
sum_of_loss = 0


for iters in range(max_iter):
  imgo,seg,_ = dataset.Train.next_image()
  img = normalizer(transforms.functional.to_tensor(imgo)).unsqueeze(0)
  seg = transforms.functional.to_tensor(seg).unsqueeze(0)
  img = Variable(img.cuda())
  seg = Variable(seg.cuda()) 
  
  optimizer.zero_grad()
  y_pred = model(img)
  loss = loss_function(y_pred,seg)
  sum_of_loss += loss.item()
  loss.backward()
  optimizer.step()
  scheduler.step(loss)

  if (iters+1)%25 == 0:
    print ("Iteration = {:0>3d}, Optimizer = Adam, Loss = {:.5f}".format(iters+1,sum_of_loss/(iters+1)))
  
  if (iters+1)% 100 == 0:
    plt.subplot(1,3,1,label="Input")
    plt.imshow(imgo)
    plt.subplot(1,3,2,label="Ground Truth")
    plt.imshow(transforms.functional.to_pil_image(seg.cpu().squeeze(0)))
    seg_pil = transforms.functional.to_pil_image(seg.cpu().squeeze(0))
    seg_np = np.array(seg_pil)
    np.save(f'/content/drive/MyDrive/tinyunet/groundtruthtrainsave/seg_{iters}.npy', seg_np)
    plt.subplot(1,3,3,label="Model Output")
    plt.imshow(transforms.functional.to_pil_image(y_pred.cpu().squeeze(0)))
    y_pred_pil = transforms.functional.to_pil_image(y_pred.cpu().squeeze(0))
    y_pred_np = np.array(y_pred_pil)
    np.save(f'/content/drive/MyDrive/tinyunet/groundtruthtrainsave/y_pred_{iters}.npy', y_pred_np)
    plt.show()
    time.sleep(0.5)
  
torch.save(model.state_dict(),"/content/drive/MyDrive/tinyunet/modelsave/drive_aaf14x16_exp1Fulladamtanpaaugmen.mdl")
