import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class Multihead_self_attention(nn.Module):

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            attn_mask: torch.Tensor = None,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.attn_mask = attn_mask

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.attn_mask is not None:
            x = F.scaled_dot_product_attention(
                q, k, v, attn_mask=self.attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )

        else:
            x = F.scaled_dot_product_attention(
                q, k, v, 
                dropout_p=self.attn_drop.p if self.training else 0.,
            )

        x = x.transpose(1, 2).reshape(B, L, D)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Multihead_cross_attention(nn.Module):

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5



        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)


    def forward(self, q, k, v) -> torch.Tensor:
        B, L, D = q.shape
        _, L_k, _ = k.shape
        _, L_v, _ = v.shape
        q = self.q_proj(q).reshape(B, self.num_heads, L, self.head_dim)
        k = self.k_proj(k).reshape(B, self.num_heads, L_k, self.head_dim)
        v = self.v_proj(v).reshape(B, self.num_heads, L_v, self.head_dim)
        q, k = self.q_norm(q), self.k_norm(k)

        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.,
        )

        x = x.transpose(1, 2).reshape(B, L, D)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


#################################################################################
#         Embedding layers for Timesteps, Labels and time-series                #
#################################################################################

class CausalConv1d(torch.nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):

        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias)
        
        self.__padding = (kernel_size - 1) * dilation
        
    def forward(self, input):
        return super(CausalConv1d, self).forward(F.pad(input, (self.__padding, 0)))

class DilatedCNN(nn.Module):
    
    def __init__(self,in_channels, out_channels, depth=5):
        super().__init__()

        self.dil1 = CausalConv1d(in_channels=in_channels,out_channels=2*out_channels,kernel_size=2,
                                dilation=1)
        self.dil2 = CausalConv1d(in_channels=2*out_channels,out_channels=2*out_channels,kernel_size=2,
                                dilation=2)
        self.dil3 = CausalConv1d(in_channels=2*out_channels,out_channels=2*out_channels,kernel_size=2,
                                dilation=4)
        self.dil4 = CausalConv1d(in_channels=2*out_channels,out_channels=out_channels,kernel_size=2,
                                dilation=8)
        self.dil5 = CausalConv1d(in_channels=out_channels,out_channels=out_channels,kernel_size=2,
                                dilation=16)
        
        self._init_weights()
    
    def _init_weights(self):
        for layer in [self.dil1, self.dil2, self.dil3, self.dil4, self.dil5]:
            nn.init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        
        # for predicting y0(y at timestep 0), we need to use
        # x-value at the last timestep.
        
        x = torch.relu(self.dil1(x))
        x = torch.relu(self.dil2(x))
        x = torch.relu(self.dil3(x))
        x = torch.relu(self.dil4(x))
        x = torch.relu(self.dil5(x))
        
        return x



if __name__ == '__main__':
    x = torch.from_numpy(np.ones((3, 2, 5))).float()
    B,C,L = x.shape
    hidden_size = 16
    dialatedCCN = DilatedCNN(in_channels=C, out_channels=hidden_size)
    out = dialatedCCN(x)
    print(out.size())
