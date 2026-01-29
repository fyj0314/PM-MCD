import torch
from torch import nn
import torch.nn.functional as F
from models.vmamba import Backbone_VSSM
from models.MSSCBlock import MSSCBlock

class Conv3(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, padding=1):
        super(Conv3, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, img):
        x = self.conv(img)
        x = self.norm(x)
        x = self.relu(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class MLP_Upsample(nn.Module):
    def __init__(self, Cconc, mlp_ratio=4, dims=(64, 128, 320, 512)):
        super(MLP_Upsample, self).__init__()
        self.mlp1 = Mlp(in_features=dims[0], hidden_features=dims[0]*mlp_ratio, out_features=Cconc)
        self.mlp2 = Mlp(in_features=dims[1], hidden_features=dims[1]*mlp_ratio, out_features=Cconc)
        self.mlp3 = Mlp(in_features=dims[2], hidden_features=dims[2]*mlp_ratio, out_features=Cconc)
        self.mlp4 = Mlp(in_features=dims[3], hidden_features=dims[3]*mlp_ratio, out_features=Cconc)

    def forward(self, f1, f2, f3, f4):
        size = (f1.shape[2], f1.shape[3])
        f1 = self.mlp1(f1.permute(0, 2, 3, 1))
        f2 = self.mlp2(f2.permute(0, 2, 3, 1))
        f2 = F.interpolate(f2.permute(0, 3, 1, 2), size, mode='bilinear', align_corners=True)
        f3 = self.mlp3(f3.permute(0, 2, 3, 1))
        f3 = F.interpolate(f3.permute(0, 3, 1, 2), size, mode='bilinear', align_corners=True)
        f4 = self.mlp4(f4.permute(0, 2, 3, 1))
        f4 = F.interpolate(f4.permute(0, 3, 1, 2), size, mode='bilinear', align_corners=True)

        return torch.cat((f1, f2.permute(0, 2, 3, 1), f3.permute(0, 2, 3, 1), f4.permute(0, 2, 3, 1)), dim=3)

class PMMCD(nn.Module):
    def __init__(self, Cconc=256, n_class=10, **kwargs):
        super().__init__()
        self.encoder = Backbone_VSSM(out_indices=(0, 1, 2, 3), **kwargs)
        self.dims = self.encoder.dims

        self.diff1 = Conv3(self.dims[0]*2, self.dims[0])
        self.diff2 = Conv3(self.dims[1]*2, self.dims[1])
        self.diff3 = Conv3(self.dims[2]*2, self.dims[2])
        self.diff4 = Conv3(self.dims[3]*2, self.dims[3])
        self.mlp_up = MLP_Upsample(Cconc=Cconc, dims=self.dims)
        self.mlp = Mlp(in_features=4*Cconc, hidden_features=4*Cconc*4, out_features=Cconc)
        self.mssc = MSSCBlock(Cconc, Cconc)
        self.cout = nn.Conv2d(in_channels=Cconc, out_channels=n_class, kernel_size=3, stride=1, padding=1)

    def forward(self, t0, t1):
        F_T0 = self.encoder(t0)
        F_T1 = self.encoder(t1)
        F1_T0, F2_T0, F3_T0, F4_T0 = F_T0
        F1_T1, F2_T1, F3_T1, F4_T1 = F_T1
        F1_dist = self.diff1(torch.cat((F1_T0, F1_T1), dim=1))
        F2_dist = self.diff2(torch.cat((F2_T0, F2_T1), dim=1))
        F3_dist = self.diff3(torch.cat((F3_T0, F3_T1), dim=1))
        F4_dist = self.diff4(torch.cat((F4_T0, F4_T1), dim=1))
        f = self.mlp_up(F1_dist, F2_dist, F3_dist, F4_dist)
        f = self.mlp(f)
        f = self.mssc(f.permute(0, 3, 1, 2))
        f = F.interpolate(f, scale_factor=4, mode='bilinear', align_corners=True)
        out = self.cout(f)

        return out