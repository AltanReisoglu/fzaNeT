import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class MyDataset(torch.utils.data.Dataset):
    
    def __init__(self):
        super(MyDataset, self).__init__()

    def __getitem__(self, index):
        return 

    def __len__(self):
        return 
    

dataloader = torch.utils.DataLoader(dataset, batch_size=1, suffle=False)