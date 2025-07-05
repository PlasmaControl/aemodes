import torch
import torch.nn as nn


class ASTModel(nn.Module):
    def __init__(
        self, 
        input_size=(4, 355, 128),
        output_size=(355, 5),
    ):
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        

    def forward(self, x):
        x = self.features(x)  # [B, C, W, H]
        
        B, C, W, H = x.shape
        x = x.view(B*W, C*H)  # Flatten for classification
        x = self.classifier(x)
        
        x = x.view(B, W, -1)  # Reshape to [B, W, output_size[1]]
        return x

if __name__ == "__main__":
    import torchsummary
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    input_size = (4, 355, 128)
    model = ASTModel().to(device)
    torchsummary.summary(model, input_size=input_size)
    input_tensor = torch.randn(2, *input_size).to(device)
    with torch.no_grad(): output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)