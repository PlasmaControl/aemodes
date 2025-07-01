import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(
        self, 
        input_size = (4, 355, 128),
        output_size = (355, 5),
        kernel_size = 3, 
        channels = [4, 8, 16, 32],
        linear_hidden = 16,
        dropout_p = [0.5, 0.5, 0.5, 0.5]
        ):
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        
        features = []
        for i in range(len(channels)-1):
            features.append(nn.Conv2d(
                    in_channels=channels[i], 
                    out_channels=channels[i+1], 
                    kernel_size=kernel_size, 
                    stride=1, 
                    padding=1,
                    ))
            features.append(nn.BatchNorm2d(channels[i+1]))
            features.append(nn.ReLU(inplace=True))
            features.append(nn.MaxPool2d(                   # double check max pooling inconsistent size
                kernel_size=(1,2), 
                stride=1,
                ))
            features.append(nn.Dropout(dropout_p[i]))
        self.features = nn.Sequential(*features)

        self.classifier = nn.Sequential(
            nn.Linear(4000, linear_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p[-1]),
            nn.Linear(linear_hidden, self.output_size[1])   # logits
        )

    def forward(self, x):
        x = self.features(x) # [B, 32, 355, 128]
        B, C, W, H = x.size() # B = Batch, C = Channels, W = Width (Time), H = Height (Freq)
        x = x.view(B*W, C*H) # do this on spectrogram itself to see if it combines correctly
        
        x = self.classifier(x)
        x = x.view(B, W, -1)
        return x
    
if __name__ == "__main__":
    import torchsummary
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = Model().to(device)
    input_size = (4, 355, 128)
    input_tensor = torch.randn(2, *input_size).to(device)
    torchsummary.summary(model, input_size=input_size)
    output = model(input_tensor)
    print("Output shape:", output.shape)