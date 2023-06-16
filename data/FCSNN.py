import torch
import torch.nn as nn
import torch.nn.functional as F


class SiameseNet(nn.Module):
    def __init__(self, input_nbr, label_nbr):
      super(SiameseNet, self).__init__()
      self.input_nbr = input_nbr
      self.label_nbr = label_nbr
      cur_depth = input_nbr
      # Define the shared convolutional backbone
      self.prima_sezione = nn.Sequential(
          # inizio primo blocco
          nn.Conv2d(input_nbr, 16, kernel_size=3, padding=1),
          nn.BatchNorm2d(16),
          nn.ReLU(inplace=True),
          nn.Conv2d(16, 16, kernel_size=3, padding=1),
          nn.BatchNorm2d(16),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=2,stride=1),      #inizio secondo blocco
          nn.Conv2d(16, 32, kernel_size=3,padding=1),
          nn.BatchNorm2d(32),
          nn.ReLU(inplace=True),
          nn.ConvTranspose2d(32, 16, kernel_size=3,padding=1),
          nn.BatchNorm2d(16),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=2,stride=1,padding=1),
          #inizio terzo blocco
          nn.Conv2d(16, 32, kernel_size=3,padding=1),
          nn.BatchNorm2d(32),
          nn.ReLU(inplace=True),
          nn.Conv2d(32, 16, kernel_size=3,padding=1),
          nn.BatchNorm2d(16),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=2,stride=1,padding=1),
          #inizio quarto blocco
          nn.Conv2d(16, 16, kernel_size=3),
          nn.BatchNorm2d(16),
          nn.ReLU(inplace=True),
          nn.Conv2d(16, 8, kernel_size=3, padding=1),
          nn.BatchNorm2d(8),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=2,stride=1,padding=1),
          #inizio quinto blocco, da inserire delation!!!!
          nn.Conv2d(8, 16, kernel_size=3, padding=1),
          nn.BatchNorm2d(16),
          nn.ReLU(inplace=True),
          nn.Conv2d(16, 8, kernel_size=3,  padding=1),
          nn.BatchNorm2d(8),
          nn.ReLU(inplace=True),
          nn.Conv2d(8, 16, kernel_size=3, padding=1),
          nn.BatchNorm2d(16),
          nn.ReLU(inplace=True),
      )
      self.seconda_sezione = nn.Sequential(
          nn.Conv2d(4, 16, kernel_size=3,padding=1),
          nn.BatchNorm2d(16),
          nn.ReLU(inplace=True),
          nn.Conv2d(16, 8, kernel_size=3,padding=1),
          nn.BatchNorm2d(8),
          nn.ReLU(inplace=True),
      )
      #da mettere logsoftmax, è interessante. era presente una batchNorm
      self.ultima_sezione = nn.Sequential(
          nn.Conv2d(32, 64, kernel_size=3,padding=1),
          nn.ReLU(inplace=True),
      )
      # Define the fully connected layers for computing the similarity score, era presente una batchNorm
      self.fc = nn.Sequential(
                nn.Conv2d(64, 16, kernel_size=3,padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Conv2d(16,label_nbr, kernel_size=3, padding=1),
                nn.LogSoftmax(dim=1),
            )








    
    def forward_once(self, x):
      x = self.prima_sezione(x)
      return x
    
    def forward_twice(self,x1,x2,x3,x4):
        x1=self.seconda_sezione(x1)
        x2=self.seconda_sezione(x2)
        x3=self.seconda_sezione(x3)
        x4=self.seconda_sezione(x4)
        return torch.cat([x1, x2, x3, x4], dim=1)

    def forward(self, x1, x2):
        #rimettere label
        """Forward method."""
        out1=self.forward_once(x1)
        out2=self.forward_once(x2)
        #loss1 = FocalContrastiveLoss(out1,out2,label)
        out11,out12,out13,out14 = torch.chunk(out1, 4, dim=1)
        out21,out22,out23,out24 = torch.chunk(out2, 4, dim=1)
        out1=self.forward_twice(out11,out12,out13,out14)
        out2=self.forward_twice(out21,out22,out23,out24)
        #loss2 = FocalContrastiveLoss(out1,out2,label)
        out1=self.ultima_sezione(out1)
        out2=self.ultima_sezione(out2)
        #loss3 = FocalContrastiveLoss(out1,out2,label)
        #out1 = out1.view(out1.size(0), -1)
        #out2 = out2.view(out2.size(0), -1) #possibile scelta, mettere la threshold alla fine come risultato!
        #threshold=0.5
        out=torch.abs(out1-out2)
        #out=torch.threshold(out,threshold,0)
        #out=torch.exp(-torch.sum(torch.abs(out1 - out2), keepdims=True))
        out = self.fc(out)
        return out