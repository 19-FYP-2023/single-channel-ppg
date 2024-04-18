import torch 
from torch import nn
import torch.nn.functional as F
import math
import numpy as np


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, first_stride, skip_stride, mid_channels=None, second_stride=None, initial_block=False, kernel_size=3, first_padding=2, second_padding=2, skip_padding=1):
        super().__init__()

        if not mid_channels:
            mid_channels = out_channels


        if initial_block:
            self.double_conv = nn.Sequential(
                nn.Conv1d(in_channels, mid_channels, kernel_size=kernel_size, stride=first_stride, padding=first_padding), 
                nn.BatchNorm1d(mid_channels),
                nn.ReLU(inplace=False),
                nn.Conv1d(mid_channels, out_channels, kernel_size=kernel_size, stride=second_stride, padding=second_padding)
            )

        else: 
            self.double_conv = nn.Sequential(
                nn.BatchNorm1d(in_channels),
                nn.ReLU(inplace=False),
                nn.Conv1d(in_channels, mid_channels, kernel_size=kernel_size, stride=first_stride, padding=first_padding), 
                nn.BatchNorm1d(mid_channels),
                nn.ReLU(inplace=False),
                nn.Conv1d(mid_channels, out_channels, kernel_size=kernel_size, stride=second_stride, padding=second_padding)
            )

        self.skip_block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=skip_stride, padding=skip_padding), 
            nn.BatchNorm1d(out_channels)
        )


    def forward(self, x):
        return self.double_conv(x) + self.skip_block(x)
    

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, first_stride, mid_channels=None, second_stride=None, kernel_size=3, first_padding=1, second_padding=1):
        super().__init__()

        if not mid_channels:
            mid_channels = out_channels
 
        self.double_conv = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=False),
            nn.Conv1d(in_channels, mid_channels, kernel_size=kernel_size, stride=first_stride, padding=first_padding), 
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=False),
            nn.Conv1d(mid_channels, out_channels, kernel_size=kernel_size, stride=second_stride, padding=second_padding)
        )

    def forward(self, x):
        return self.double_conv(x)
    

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.conv_transpose =   nn.ConvTranspose1d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=2, stride=self.scale_factor)
        self.batch_norm = nn.BatchNorm1d(out_channels)


    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.batch_norm(x)
        return x
    

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_size, embedding_length):
        super().__init__()

        self.embedding_size = embedding_size
        self.embedding_length = embedding_length

        pe = torch.zeros(embedding_size, embedding_length)
        position = torch.arange(0, embedding_size).unsqueeze(1)
        div_term = 1/torch.pow(10000, torch.arange(0, embedding_length, 2)/embedding_length)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer(f"pe", pe)

### FNet blocks
class FNetFeedForward(nn.Module):
    def __init__(self, dhidden):
        super().__init__()
        self.dense_1 = nn.Linear(dhidden, 4*dhidden) #d_feedforward is 4*d_model
        self.dense_2 = nn.Linear(4*dhidden, dhidden)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = self.dense_1(x)
        x = self.gelu(x)
        x = self.dense_2(x)
        x = self.dropout(x)
        return x

class FourierAttentionBlock(nn.Module):
    def __init__(self, dhidden=512):
        super().__init__()
        self.LayerNorm1 = nn.LayerNorm(dhidden)
        self.feedforward = FNetFeedForward(dhidden)
        self.LayerNorm2 = nn.LayerNorm(dhidden)

    def forward(self, x):

        # DFT along hidden dimension followed by DFT along sequence dimension. Only keep real part of the result
        x_fft = torch.real(torch.fft.fft(torch.fft.fft(x).T).T)
        x = self.LayerNorm1(x_fft)

        x_ff = self.feedforward(x)
        x = self.LayerNorm2(x_ff)
        return x
    

class FeedForwardLayer(nn.Module):
    def __init__(self):
        super().__init__()

        self.ffn = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Linear(1024, 1024)
        )

    def forward(self, x):
        return self.ffn(x)


class PPGFNet(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.in_channels = in_channels

        self.down_conv_layer1 = DoubleConv(in_channels=in_channels, out_channels=32, first_stride=1, second_stride=1, skip_stride=1, initial_block=True, first_padding=1, second_padding=1)
        self.down_conv_layer2 = DoubleConv(in_channels=32, out_channels=64, first_stride=2, second_stride=1, skip_stride=2, first_padding=1, second_padding=1)
        self.down_conv_layer3 = DoubleConv(in_channels=64, out_channels=128, first_stride=2, second_stride=1, skip_stride=2, first_padding=1, second_padding=1)
        self.down_conv_layer4 = DoubleConv(in_channels=128, out_channels=256, first_stride=2, second_stride=1, skip_stride=2, first_padding=1, second_padding=1)
        self.down_conv_layer5 = DoubleConv(in_channels=256, out_channels=512, first_stride=2, second_stride=1, skip_stride=2, first_padding=1, second_padding=1)
        self.bottleneck_layer = Bottleneck(in_channels=512, out_channels=512, first_stride=1, second_stride=1, first_padding=1, second_padding=1)

        self.posencoding1 = PositionalEncoding(embedding_size=32, embedding_length=1024)
        self.posencoding2 = PositionalEncoding(embedding_size=64, embedding_length=512)
        self.posencoding3 = PositionalEncoding(embedding_size=128, embedding_length=256)
        self.posencoding4 = PositionalEncoding(embedding_size=256, embedding_length=128)
        self.posencodingfinal = PositionalEncoding(embedding_size=32, embedding_length=1024)

        self.attn_layer1 = FourierAttentionBlock(dhidden=1024)
        self.attn_layer2 = FourierAttentionBlock(dhidden=512)
        self.attn_layer3 = FourierAttentionBlock(dhidden=256)
        self.attn_layer4 = FourierAttentionBlock(dhidden=128)
        self.attn_layerfinal = FourierAttentionBlock(dhidden=1024)

        self.up_conv_layer1 = DoubleConv(in_channels=768, out_channels=256, first_stride=1, second_stride=1, skip_stride=1, first_padding=1, second_padding=1)
        self.up_conv_layer2 = DoubleConv(in_channels=384, out_channels=128, first_stride=1, second_stride=1, skip_stride=1, first_padding=1, second_padding=1)
        self.up_conv_layer3 = DoubleConv(in_channels=192, out_channels=64, first_stride=1, second_stride=1, skip_stride=1, first_padding=1, second_padding=1)
        self.up_conv_layer4 = DoubleConv(in_channels=96, out_channels=32, first_stride=1, second_stride=1, skip_stride=1, first_padding=1, second_padding=1)

        self.upsampling_layer1 = UpSample(in_channels=512, out_channels=512, scale_factor=2)
        self.upsampling_layer2 = UpSample(in_channels=256, out_channels=256, scale_factor=2)
        self.upsampling_layer3 = UpSample(in_channels=128, out_channels=128, scale_factor=2)
        self.upsampling_layer4 = UpSample(in_channels=64, out_channels=64, scale_factor=2)

        self.feedforward = FeedForwardLayer()

    def forward(self,x):

        down_conv1 = self.down_conv_layer1(x)
        down_conv2 = self.down_conv_layer2(down_conv1)
        down_conv3 = self.down_conv_layer3(down_conv2)
        down_conv4 = self.down_conv_layer4(down_conv3)
        down_conv5 = self.down_conv_layer5(down_conv4)
        bottleneck = self.bottleneck_layer(down_conv5)

        posencoded_conv1 = self.posencoding1.pe + down_conv1
        posencoded_conv2 = self.posencoding2.pe + down_conv2
        posencoded_conv3 = self.posencoding3.pe + down_conv3
        posencoded_conv4 = self.posencoding4.pe + down_conv4

        attn1 = self.attn_layer1(posencoded_conv1, posencoded_conv1, posencoded_conv1)
        attn2 = self.attn_layer2(posencoded_conv2, posencoded_conv2, posencoded_conv2)
        attn3 = self.attn_layer3(posencoded_conv3, posencoded_conv3, posencoded_conv3)
        attn4 = self.attn_layer4(posencoded_conv4, posencoded_conv4, posencoded_conv4)
        

        upsampling1 = self.upsampling_layer1(bottleneck)
        concat1 = torch.cat((upsampling1 , attn4), dim=1)
        upconv_1  = self.up_conv_layer1(concat1)

        upsampling2 = self.upsampling_layer2(upconv_1)
        concat2 = torch.cat((upsampling2, attn3), dim=1)
        upconv_2 = self.up_conv_layer2(concat2)

        upsampling3 = self.upsampling_layer3(upconv_2)
        concat3 = torch.cat((upsampling3, attn2), dim=1)
        upconv_3 = self.up_conv_layer3(concat3)

        upsampling4 = self.upsampling_layer4(upconv_3)
        concat4 = torch.cat((upsampling4, attn1), dim=1)
        upconv_4 = self.up_conv_layer4(concat4)


        posencoded_final = self.posencodingfinal.pe + upconv_4
        attnfinal = self.attn_layerfinal(posencoded_final, posencoded_final, posencoded_final)

        attnout = attnfinal[:, 0]

        out = self.feedforward(attnout)

        out = out.unsqueeze(1)
        return out