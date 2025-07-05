import torch
import torch.nn as nn

class BCELoss(nn.BCEWithLogitsLoss):
    """Binary Cross Entropy Loss with logits."""
    def __init__(
        self, **kwargs):
        super().__init__(**kwargs)
        
    def forward(self, input, target):
        return super().forward(input, target.float())
    
class DiceLoss(nn.CrossEntropyLoss):
    """Cross Entropy Loss for multi-class classification."""
    def __init__(
        self, **kwargs):
        super().__init__(**kwargs)
        
    def forward(self, input, target):
        return super().forward(input, target.long())
    
class MSELoss(nn.MSELoss):
    """Mean Squared Error Loss."""
    def __init__(
        self, **kwargs):
        super().__init__(**kwargs)
        
    def forward(self, input, target):
        input = torch.sigmoid(input)
        return super().forward(input, target.float())
    
class MAELoss(nn.Module):
    """Mean Absolute Error Loss."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def forward(self, input, target):
        input = torch.sigmoid(input)
        return torch.mean(torch.abs(input - target.float()))  # Average over the batch
    
class TPRLoss(nn.Module):
    """True Positive Rate Loss."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def forward(self, input, target):
        input = torch.sigmoid(input)
        tp = (input * target).sum()
        fn = ((1 - input) * target).sum()
        tpr = tp / (tp + fn + 1e-8)  # Avoid division by zero
        return 1 - tpr  # We want to maximize TPR, so we minimize 1 - TPR
    
class FPRLoss(nn.Module):
    """False Positive Rate Loss."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def forward(self, input, target):
        input = torch.sigmoid(input)
        fp = (input * (1 - target)).sum()
        tn = ((1 - input) * (1 - target)).sum()
        fpr = fp / (fp + tn + 1e-8)  # Avoid division by zero
        return fpr  # We want to minimize FPR
    
class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance."""
    def __init__(self, alpha=0.25, gamma=2.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, input, target):
        input = torch.sigmoid(input)
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(input, target.float())
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()  # Average over the batch