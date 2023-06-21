import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, first_block, encoder, bias):
        super(ResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel = kernel
        self.stride = stride
        self.first_block = first_block
        self.bias = bias
        self.encoder = encoder
        
        if self.encoder:
            self.conv1 = nn.Conv2d(in_channels=self.in_channels,
                                   out_channels=self.out_channels, 
                                   kernel_size=self.kernel, 
                                   stride=self.stride, 
                                   padding=1, 
                                   bias=self.bias)
            self.conv2 = nn.Conv2d(in_channels=self.out_channels,
                                   out_channels=self.out_channels, 
                                   kernel_size=self.kernel, 
                                   stride=1, 
                                   padding=1,  
                                   bias=self.bias)
            self.input_skip = nn.Conv2d(in_channels=self.in_channels, 
                                        out_channels=self.out_channels, 
                                        kernel_size=1,
                                        stride=self.stride, 
                                        padding=0,
                                        bias=self.bias)
        else:
            self.conv1 = nn.Conv2d(in_channels=self.in_channels,
                                out_channels=self.out_channels, 
                                kernel_size=self.kernel, 
                                stride=self.stride, 
                                padding=1, 
                                bias=self.bias)
            self.conv2 = nn.Conv2d(in_channels=self.out_channels,
                                out_channels=self.out_channels, 
                                kernel_size=self.kernel, 
                                stride=self.stride,
                                padding=1,
                                bias=self.bias)
            self.input_skip = nn.Conv2d(in_channels=self.in_channels, 
                                        out_channels=self.out_channels, 
                                        kernel_size=1, 
                                        stride=1, 
                                        padding=0,
                                        bias=self.bias)

        if first_block:
            self.bn1 = nn.BatchNorm2d(self.out_channels)
        else:
            self.bn1 = nn.BatchNorm2d(self.in_channels)
            self.bn2 = nn.BatchNorm2d(self.out_channels)

        self.activation1 = nn.ReLU(inplace=True)
        self.activation2 = nn.ReLU(inplace=True)
         
    def forward(self, x):
        if self.first_block:
            output = self.conv1(x)
            output = self.activation1(self.bn1(output))
            output = self.conv2(output)
            output = output + self.input_skip(x) 
            return output
        else:
            output = self.activation1(self.bn1(x))
            output = self.conv1(output)
            output = self.activation2(self.bn2(output))
            output = self.conv2(output)
            output = output + self.input_skip(x)
            return output

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, dilation, bias):
        super(ASPP, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dilation = dilation
        self.bias = bias

        self.aspp1 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, 
                      out_channels=self.out_channels, 
                      kernel_size=1, 
                      stride=1,
                      padding=0, 
                      dilation=self.dilation[0],
                      bias=self.bias),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True)
        )

        self.aspp2 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, 
                      out_channels=self.out_channels, 
                      kernel_size=3, 
                      stride=1,
                      padding=6, 
                      dilation=self.dilation[1],
                      bias=self.bias),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True)
        )

        self.aspp3 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, 
                      out_channels=self.out_channels, 
                      kernel_size=3, 
                      stride=1,
                      padding=12, 
                      dilation=self.dilation[2],
                      bias=self.bias),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True)
        )
        self.aspp4 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, 
                      out_channels=self.out_channels, 
                      kernel_size=3, 
                      stride=1,
                      padding=18, 
                      dilation=self.dilation[3],
                      bias=self.bias),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True)
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.out_channels*4, out_channels=self.out_channels, kernel_size=1, bias=self.bias),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        aspp1 = self.aspp1(x)
        aspp2 = self.aspp2(x)
        aspp3 = self.aspp3(x)
        aspp4 = self.aspp4(x)
        output = torch.cat((aspp1, aspp2, aspp3, aspp4), dim=1)   
        output = self.conv1(output)
        return output
    
class SelfAttention(nn.Module):
    def __init__(self, in_channels, out_channels, ratio, bias):
        super(SelfAttention, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ratio = ratio
        self.bias  = bias
        self.key   = nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels//self.ratio, kernel_size=1, bias=self.bias)
        self.query = nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels//self.ratio, kernel_size=1, bias=self.bias)
        self.value = nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels//self.ratio, kernel_size=1, bias=self.bias)
        self.attention_conv = nn.Conv2d(in_channels=self.in_channels//self.ratio, out_channels=self.out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        key   = self.key(x)
        query = self.query(x)
        value = self.value(x)
        key_dimension = torch.tensor(key.shape[-1])

        key = torch.transpose(key,2,3)

        numerator = query*key
        denominator = torch.sqrt(key_dimension)
        output = F.softmax(numerator/denominator)*value
        output = self.attention_conv(output)
        return output
     
class SqueezeExcite(nn.Module):
    def __init__(self, channels, r):
        super(SqueezeExcite, self).__init__()
        self.channels = channels
        self.r = r

        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excite  = nn.Sequential(
            nn.Linear(in_features=self.channels, out_features=self.channels//self.r, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=self.channels//self.r, out_features=self.channels, bias=True),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        output = self.squeeze(x).view(b,c)
        output = self.excite(output).view(b, c, 1, 1)
        output = output * x
        return output
    
class ResUnetPP(nn.Module):
    def __init__(self, in_channels: int, encoder_out_channels: List[int], class_channels: int, se_ratio: int, attention_ratio: int, dilation: List[int], bias: bool):
        '''
        Examples:
        in_channels = 3
        encoder_out_channels = [16,32,64,128] {maximum 4 numbers}
        class_channels = 10
        se_ratio = 8                          {choose one of the following sequence 2, 4, 8, 16}
        attention_ratio = 8                   {choose one of the following sequence 2, 4, 8, 16}
        dilation = [1,6,12,18]                {maximum 4 numbers}
        '''
        super(ResUnetPP, self).__init__()
        self.in_channels          = in_channels
        self.encoder_out_channels = encoder_out_channels
        self.class_channels       = class_channels
        self.se_ratio             = se_ratio
        self.attention_ratio      = attention_ratio
        self.dilation             = dilation
        self.bias                 = bias

        self.attention_channels1 = self.encoder_out_channels[3]*2                          # 256
        self.attention_channels2 = self.attention_channels1 + self.encoder_out_channels[2] # 256+64 = 320
        self.attention_channels3 = self.attention_channels2//2                             # 160
        self.attention_channels4 = self.attention_channels3 + self.encoder_out_channels[1] # 160+32 = 192
        self.attention_channels5 = self.attention_channels4//2                             # 96
        self.attention_channels6 = self.attention_channels5 + self.encoder_out_channels[0] # 96+16  = 112
       
        # Encoder
        self.encoder_block1 = ResBlock(in_channels=self.in_channels,             out_channels=self.encoder_out_channels[0], kernel=3, stride=1, first_block=True,  bias=self.bias, encoder=True)
        self.encoder_block2 = ResBlock(in_channels=self.encoder_out_channels[0], out_channels=self.encoder_out_channels[1], kernel=3, stride=2, first_block=False, bias=self.bias, encoder=True)
        self.encoder_block3 = ResBlock(in_channels=self.encoder_out_channels[1], out_channels=self.encoder_out_channels[2], kernel=3, stride=2, first_block=False, bias=self.bias, encoder=True)
        self.encoder_block4 = ResBlock(in_channels=self.encoder_out_channels[2], out_channels=self.encoder_out_channels[3], kernel=3, stride=2, first_block=False, bias=self.bias, encoder=True)

        self.se1 = SqueezeExcite(channels=self.encoder_out_channels[0], r=self.se_ratio) # 16
        self.se2 = SqueezeExcite(channels=self.encoder_out_channels[1], r=self.se_ratio) # 32
        self.se3 = SqueezeExcite(channels=self.encoder_out_channels[2], r=self.se_ratio) # 64

        # Bridge
        self.aspp1 = ASPP(in_channels=self.encoder_out_channels[3], out_channels=self.encoder_out_channels[3]*2, dilation=self.dilation, bias=self.bias)

        # Decoder
        self.attention1     = SelfAttention(in_channels=self.attention_channels1, out_channels=self.attention_channels1, ratio=self.attention_ratio, bias=self.bias)
        self.upsample1      = F.interpolate
        self.decoder_block1 = ResBlock(in_channels=self.attention_channels2, out_channels=self.attention_channels2//2, kernel=3, stride=1, first_block=False, bias=self.bias, encoder=False)

        self.attention2     = SelfAttention(in_channels=self.attention_channels3, out_channels=self.attention_channels3, ratio=self.attention_ratio, bias=self.bias)
        self.upsample2      = F.interpolate
        self.decoder_block2 = ResBlock(in_channels=self.attention_channels4, out_channels=self.attention_channels4//2, kernel=3, stride=1, first_block=False, bias=self.bias, encoder=False)

        self.attention3     = SelfAttention(in_channels=self.attention_channels5, out_channels=self.attention_channels5, ratio=self.attention_ratio, bias=self.bias)
        self.upsample3      = F.interpolate
        self.decoder_block3 = ResBlock(in_channels=self.attention_channels6, out_channels=self.attention_channels6//2, kernel=3, stride=1, first_block=False, bias=self.bias, encoder=False)

        # Classifier
        self.aspp2 = ASPP(in_channels=self.attention_channels6//2, out_channels=self.attention_channels6, dilation=self.dilation, bias=self.bias)
        self.out   = nn.Conv2d(in_channels=self.attention_channels6, out_channels=self.class_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        a = x = self.encoder_block1(x) 
        x = self.se1(x)                
        b = x = self.encoder_block2(x)
        x = self.se2(x)
        c = x = self.encoder_block3(x)
        x = self.se3(x)
        x = self.encoder_block4(x) 

        # Bridge
        x = self.aspp1(x)

        # Decoder
        x = self.attention1(x) 
        x = self.upsample1(x, scale_factor=2, mode='bilinear', align_corners=True) 
        x = torch.cat((c, x), dim=1) 
        x = self.decoder_block1(x) 

        x = self.attention2(x)
        x = self.upsample2(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat((b, x), dim=1)
        x = self.decoder_block2(x)

        x = self.attention3(x)
        x = self.upsample3(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat((a, x), dim=1)
        x = self.decoder_block3(x) 

        # Classifier
        x = self.aspp2(x)
        x = self.out(x)
        return x