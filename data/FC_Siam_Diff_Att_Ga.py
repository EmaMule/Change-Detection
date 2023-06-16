import torch
import torch.nn as nn

class FC_Siam_diff_Att_GA(nn.Module):
    def __init__(self,input_nbr=13,label_nbr=2):
        super(FC_Siam_diff_Att_GA, self).__init__()
        self.input_nbr = input_nbr
        self.label_nbr = label_nbr
        # Definizione degli strati della rete
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(self.input_nbr, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.diff_pooling = nn.AdaptiveAvgPool2d((1, 1))
        
        self.attention = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 12 * 12), 
            nn.Sigmoid()
        )
        
        self.global_avg_pooling = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc = nn.Sequential(
          nn.Linear(512, 2),
          nn.LogSoftmax(dim=1)  
          )# Output con due classi
        
    def forward(self, x1, x2):
        # Estrazione delle caratteristiche
        feat1 = self.feature_extractor(x1)
        feat2 = self.feature_extractor(x2)
        
        # Differenza tra le caratteristiche
        diff_feat = torch.abs(feat1 - feat2)
        
        # Aggregazione delle caratteristiche differenza
        diff_pool = self.diff_pooling(diff_feat)
        
        # Calcolo dell'attenzione
        att_weights = self.attention(diff_pool.view(diff_pool.size(0), -1))
        # Aggregazione globale delle caratteristiche
        global_feat = self.global_avg_pooling(feat2)
        
        # Moltiplicazione dell'attenzione con le caratteristiche globali
        att_feat = global_feat * att_weights
        print("feat2 size:", feat2.size())
        print("att_feat size:", att_feat.size())

        # Concatenazione delle caratteristiche
        combined_feat = torch.cat([feat2, att_feat], dim=1)
        
        # Classificazione finale
        output = self.fc(combined_feat.view(combined_feat.size(0), -1))
        
        return output
