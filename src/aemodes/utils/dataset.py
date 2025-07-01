import numpy as np
import pickle
import torch
from torch.utils.data import Dataset

class ShotDataset(Dataset):
    def __init__(self, shots, X, y):
        self.shots = shots
        self.X = X
        self.y = y
        
        lenshot = 3905
        self.nwin = 11
        self.lenwin = lenshot // self.nwin
        self.hoplen = lenshot // self.nwin
    
    def __len__(self):
        return len(self.shots) * self.nwin
    
    def __getitem__(self, idx):
        shot_idx, win_idx = idx // self.nwin, idx % self.nwin
        start_idx = win_idx * self.hoplen
        end_idx = start_idx + self.lenwin
        X = self.X[shot_idx]
        X = torch.tensor(np.stack([X['r0'],X['v1'],X['v2'],X['v3']])[:,start_idx:end_idx])
        y = torch.tensor(self.y[shot_idx][start_idx:end_idx])
        return {
            'shot': self.shots[shot_idx],
            'X': X.float(),
            'y': y.float(),
        }

def load_dataset(datapath):
    [train_shots, X_train,y_train,valid_shots, X_valid,y_valid] = pickle.load(open(datapath,'rb'))

    train_dataset = ShotDataset(train_shots, X_train, y_train)
    valid_dataset = ShotDataset(valid_shots, X_valid, y_valid)

    return train_dataset, valid_dataset