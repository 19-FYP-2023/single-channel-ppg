import torch 
from torch import nn
import torch.nn.functional as F
import math
import numpy as np


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, first_stride, skip_stride, mid_channels=None, second_stride=None, initial_block=False):
        super().__init__()

        if not mid_channels:
            mid_channels = out_channels


        if initial_block:
            self.double_conv = nn.Sequential(
                nn.Conv1d(in_channels, mid_channels, kernel_size=3, stride=first_stride, padding="same"), 
                nn.BatchNorm1d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv1d(mid_channels, out_channels, kernel_size=3, stride=second_stride, padding="same")
            )

        else: 
            self.double_conv = nn.Sequential(
                nn.BatchNorm1d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv1d(in_channels, mid_channels, kernel_size=3, stride=first_stride, padding="same"), 
                nn.BatchNorm1d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv1d(mid_channels, out_channels, kernel_size=3, stride=second_stride, padding="same")
            )

        self.skip_block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=skip_stride, padding="same"), 
            nn.BatchNorm1d(out_channels)
        )


    def forward(self, x):
        return self.double_conv(x) + self.skip_block(x)
    

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, first_stride, mid_channels=None, second_stride=None):
        super().__init__()

        if not mid_channels:
            mid_channels = out_channels
 
        self.double_conv = nn.Sequential(
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, stride=first_stride, padding="same"), 
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, stride=second_stride, padding="same")
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
    def __init__(self, embedding_size, embedding_length, level):
        super().__init__()

        self.embedding_size = embedding_size
        self.embedding_length = embedding_length

        pe = torch.zeros(embedding_size, embedding_length)
        position = torch.arange(0, embedding_size).unsqueeze(1)
        div_term = 1/torch.pow(10000, torch.arange(0, embedding_length, 2)/embedding_length)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.transpose(0, 1)
        self.register_buffer(f'pe_{level}', pe)


class AttentionWithPositionalEncoding(nn.Module):

    def __init__(self, query_dim, key_dim, value_dim, output_dim):
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.value_dim = value_dim 
        self.output_dim = output_dim 

        self.query_transform = nn.Linear(query_dim, output_dim)
        self.key_transform = nn.Linear(key_dim, output_dim)
        self.value_transform = nn.Linear(value_dim, output_dim)

    
    def forward(self, query, key, value):
        transformed_query = self.query_transform(query)
        transformed_key = self.key_transform(key)
        transformed_value = self.value_transform(value)

        attention_score = torch.matmul(transformed_query, transformed_key.transpose(0, 1))
        attention_weights = F.softmax(attention_score, dim=-1)

        output = torch.matmul(attention_weights, transformed_value)

        return output 
    

class ppgUnet(nn.Module):
    def __init__(self, in_channels, num_classes, bilinear=True):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear 

        self.down_conv1 = DoubleConv(in_channels=in_channels, out_channels=32, first_stride=1, second_stride=1, skip_stride=1, initial_block=True)
        self.down_conv2 = DoubleConv(in_channels=32, out_channels=64, first_stride=2, second_stride=1, skip_stride=2)
        self.down_conv3 = DoubleConv(inc_channels=64, out_channels=128, first_stride=2, second_stride=1, skip_stride=2)
        self.down_conv4 = DoubleConv(in_channels=128, out_channels=256, first_stride=2, second_stride=1, skip_stride=2)
        self.down_conv5 = DoubleConv(in_channels=256, out_channels=512, first_stride=2, second_stride=1, skip_stride=2)
        self.bottleneck = DoubleConv(in_channels=512, out_channels=512, first_stride=1, second_stride=1)

        self.posencoding1 = PositionalEncoding(embedding_size=32, embedding_length=1024, level=1)
        self.posencoding2 = PositionalEncoding(embedding_size=64, embedding_length=512, level=2)
        self.posencoding3 = PositionalEncoding(embedding_size=128, embedding_length=256, level=3)
        self.posencoding4 = PositionalEncoding(embedding_size=256, embedding_length=128, level=4)

        self.attn1 = AttentionWithPositionalEncoding(query_dim=1024, key_dim=1024, value_dim=1024, output_dim=1024)
        self.attn2 = AttentionWithPositionalEncoding(query_dim=512, key_dim=512, value_dim=512, output_dim=512)
        self.attn3 = AttentionWithPositionalEncoding(query_dim=256, key_dim=256, value_dim=256, output_dim=256)
        self.crossattn4 = AttentionWithPositionalEncoding(query_dim=128, key_dim=128, value_dim=128, output_dim=128)

        self.up_conv1 = DoubleConv(in_channels=768, out_channels=256, first_stride=1, second_stride=1, skip_stride=1)
        self.up_conv2 = DoubleConv(in_channels=384, out_channels=128, first_stride=1, second_stride=1, skip_stride=1)
        self.up_conv3 = DoubleConv(in_channels=192, out_channels=64, first_stride=1, second_stride=1, skip_stride=1)
        self.up_conv4 = DoubleConv(in_channels=96, out_channels=32, first_stride=1, second_stride=1, skip_stride=1)

        self.upsampling1 = UpSample(in_channels=512, out_channels=512, scale_factor=2)
        self.upsampling2 = UpSample(in_channels=256, out_channels=256, scale_factor=2)
        self.upsampling3 = UpSample(in_channels=128, out_channels=128, scale_factor=2)
        self.upsampling4 = UpSample(in_channels=64, out_channels=64, scale_factor=2)



        def forward(self,x):
            down_conv1 = self.down_conv1(x)
            down_conv2 = self.down_conv2(down_conv1)
            down_conv3 = self.down_conv3(down_conv2)
            down_conv4 = self.down_conv4(down_conv3)
            down_conv5 = self.down_conv5(down_conv4)
            bottleneck = self.bottleneck(down_conv5)

            posencoded_conv1 = self.posencoding(down_conv1)
            posencoded_conv2 = self.posencoding(down_conv2)
            posencoded_conv3 = self.posencoding(down_conv3)
            posencoded_conv4 = self.posencoding(down_conv4)

            attn1 = self.attn1(posencoded_conv1, posencoded_conv1, posencoded_conv1)
            attn2 = self.attn2(posencoded_conv2, posencoded_conv2, posencoded_conv2)
            attn3 = self.attn3(posencoded_conv3, posencoded_conv3, posencoded_conv3)
            attn4 = self.attn4(posencoded_conv4, posencoded_conv4, posencoded_conv4)

            upsampling1 = self.upsampling1(bottleneck)
            concat1 = torch.cat((upsampling1, attn4), dim=1)
            upconv_1  = self.up_conv1(concat1)

            upsampling2 = self.upsampling2(upconv_1)
            concat2 = torch.cat((upsampling2, attn3), dim=1)
            upconv_2 = self.up_conv2(concat2)

            upsampling3 = self.upsampling3(upconv_2)
            concat3 = torch.cat((upsampling3, attn2), dim=1)
            upconv_3 = self.up_conv3(concat3)

            upsampling4 = self.upsampling4(upconv_3)
            concat4 = torch.cat((upsampling4, attn1), dim=1)
            upconv_4 = self.up_conv4(concat4)

            return upconv_4 
        

        







    
