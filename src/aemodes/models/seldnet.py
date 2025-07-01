import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.segmentation import lraspp_mobilenet_v3_large
        
class SELDNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.C = 4 # the number of audio channels
        self.P = 63 # the number of feature maps
        self.Q = 3 # the number of RNN hidden units
        self.R = 32 # the dimension of first full connection
        self.K = 5 # the number of classes
        self.feature_dim = 512 # the dimension of input features
        self.cnn_layers = 2 # the number of CNN layers
        self.rnn_layers = 1 # the number of RNN layers
        ##########################################
        # CNN part
        ##########################################
        self.convs = [nn.Conv2d(in_channels=self.C,out_channels=self.P,
                        kernel_size=(3,3),padding=(1,1))]
        for _ in range(self.cnn_layers-1):
            self.convs.append(
                nn.Conv2d(in_channels=self.P,out_channels=self.P, 
                          kernel_size=(3,3),padding=(1,1)))

        # name all layers
        for i, conv in enumerate(self.convs):
            self.add_module(
                "conv{}".format(i), 
                conv)
            self.add_module(
                "bn{}".format(i), 
                nn.BatchNorm2d(self.P, affine=False))


        ##########################################
        # RNN part
        ##########################################
        self.grus = nn.GRU(
            input_size=self.P*3,
            hidden_size=self.Q,
            num_layers=self.rnn_layers,
            batch_first=True,
            bidirectional=True)
        self.add_module("grus",self.grus)


        ##########################################
        # SED part
        ##########################################
        self.sed_fc0 = nn.Linear(2*self.Q,self.R)
        self.sed_fc1 = nn.Linear(self.R,self.K)
        self.add_module("sed_fc0",self.sed_fc0)
        self.add_module("sed_fc1",self.sed_fc1)  
    
    def forward(self,x):
        # x = x.permute(0,1,3,2)
        for i in range(self.cnn_layers):
            y = F.relu(getattr(self,"conv{}".format(i))(x))
            if i<self.cnn_layers-1:
                y = F.max_pool2d(getattr(self,"bn{}".format(i))(y),kernel_size=(1,7))
            else:
                y = F.max_pool2d(getattr(self,"bn{}".format(i))(y),kernel_size=(1,5))
            x = y

        y = y.permute(0,2,3,1)
        y = y.contiguous().view(y.shape[0],y.shape[1],-1)
        y, _ = self.grus(y)
        y = torch.tanh(y)

        y = self.sed_fc1(self.sed_fc0(y))
        return y
    
if __name__ == "__main__":
    import torchsummary
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # hparams = HyperParameters()
    model = SELDNet().to(device)
    input_size = (4, 355, 128)
    input_tensor = torch.randn(1, *input_size).to(device)
    torchsummary.summary(model, input_size=input_size)
    output = model(input_tensor)
    print("Output shape:", output.shape)