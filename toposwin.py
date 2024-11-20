import torch
import torch.nn as nn
import timm
from .betti_encoder import BettiEncoder

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, y):
        b, h = x.shape[0], self.num_heads
        
        q = self.to_q(x).view(b, -1, h, x.shape[-1] // h).transpose(1, 2)
        k = self.to_k(y).view(b, -1, h, y.shape[-1] // h).transpose(1, 2)
        v = self.to_v(y).view(b, -1, h, y.shape[-1] // h).transpose(1, 2)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        x = torch.matmul(attn, v)
        x = x.transpose(1, 2).contiguous().view(b, -1, x.shape[-1] * h)
        x = self.proj(x)
        
        return x

class TopoSwin(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        

        self.swin = timm.create_model('swinv2_base_window16_256.ms_in1k',
                                      pretrained=True,
                                      features_only=True,
                                      out_indices=(0, 1, 2, 3))
        
        self.feature_dims = self.swin.feature_info.channels()

        self.b0_encoder = BettiEncoder(seq_length=100,
                                       d_model=512,
                                       nhead=4,
                                       num_layers=4,
                                       dim_feedforward=256)
        
        self.b1_encoder = BettiEncoder(seq_length=100,
                                       d_model=512,
                                       nhead=4,
                                       num_layers=4,
                                       dim_feedforward=256)
        
        self.b0_proj_layers = nn.ModuleList()
        self.b1_proj_layers = nn.ModuleList()
        self.cross_attn_b0_layers = nn.ModuleList()
        self.cross_attn_b1_layers = nn.ModuleList()
        self.norm1_layers = nn.ModuleList()
        self.norm2_layers = nn.ModuleList()
        
        for dim in self.feature_dims:
            self.b0_proj_layers.append(nn.Sequential(nn.Linear(512, dim),
                                       nn.LayerNorm(dim),
                                       nn.ReLU(),
                                       nn.Dropout(0.1)))
            
            self.b1_proj_layers.append(nn.Sequential(nn.Linear(512, dim),
                                       nn.LayerNorm(dim),
                                       nn.ReLU(),
                                       nn.Dropout(0.1)))
            
            self.cross_attn_b0_layers.append(CrossAttention(dim, num_heads=8))
            self.cross_attn_b1_layers.append(CrossAttention(dim, num_heads=8))

            self.norm1_layers.append(nn.LayerNorm(dim))
            self.norm2_layers.append(nn.LayerNorm(dim))
        
        self.num_features = self.feature_dims[-1]
        self.norm = nn.LayerNorm(self.num_features)
        self.head = nn.Linear(self.num_features, num_classes)
        
    def forward(self, x, b0, b1):
        features = self.swin(x) 
        B = x.shape[0]
        
        b0_encoded = self.b0_encoder(b0)
        b1_encoded = self.b1_encoder(b1)
        
        updated_features = []
        
        for i, feat in enumerate(features):
            H, W, C = feat.shape[1], feat.shape[2], feat.shape[3]
        
            feat = feat.reshape(B, H*W, C)  
            
            b0_proj = self.b0_proj_layers[i](b0_encoded) 
            b1_proj = self.b1_proj_layers[i](b1_encoded)
            
            feat = self.norm1_layers[i](feat)
            attn_out = self.cross_attn_b0_layers[i](feat, b0_proj)
            feat = feat + attn_out
            
            feat = self.norm2_layers[i](feat)
            attn_out = self.cross_attn_b1_layers[i](feat, b1_proj)
            feat = feat + attn_out
            
            feat = feat.reshape(B, H, W, C)
            
            updated_features.append(feat)
        

        final_feat = updated_features[-1]
        
        final_feat = final_feat.reshape(B, -1, self.num_features)
        
        x = final_feat.mean(dim=1)

        x = self.norm(x)
        x = self.head(x)
        
        return x