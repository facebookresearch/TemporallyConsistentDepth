# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn

class UNet(torch.nn.Module):
    """
    Basic UNet building block, calling itself recursively. Based on the Confidence Routing Network 
    implementation of Weder et al.'s Routed Fusion: https://github.com/weders/RoutedFusion.git
    """

    def __init__(self, cin, c0, cout, nlayers, post_activation=torch.nn.ReLU()):

        super().__init__()
        self.c0 = c0
        self.nlayers = nlayers

        self.pre = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(cin, c0, kernel_size=3, stride=1, padding=0),
            torch.nn.GroupNorm(1, c0),
            torch.nn.ReLU(),
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(c0, c0, kernel_size=3, stride=1, padding=0),
            torch.nn.GroupNorm(1, c0),
            torch.nn.ReLU()
        )
        
        self.post = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(3 * c0, c0, kernel_size=3, stride=1, padding=0),
            torch.nn.GroupNorm(1, c0),
            torch.nn.ReLU(),
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(c0, cout, kernel_size=3, stride=1, padding=0),
            torch.nn.GroupNorm(1, cout),
            post_activation,
        )

        if nlayers > 1:
            self.process = UNet(c0, 2 * c0, 2 * c0, nlayers - 1)
        else:
            self.process = torch.nn.Sequential(
                torch.nn.ReflectionPad2d(1),
                torch.nn.Conv2d(c0, 2 * c0, kernel_size=3, stride=1, padding=0),
                torch.nn.GroupNorm(1, 2 * c0),
                torch.nn.ReLU(),
                torch.nn.ReflectionPad2d(1),
                torch.nn.Conv2d(2 * c0, 2 * c0, kernel_size=3, stride=1, padding=0),
                torch.nn.GroupNorm(1, 2 * c0),
                torch.nn.ReLU()
            )

        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, data):
        features = self.pre(data)
        lower_scale = self.maxpool(features)
        lower_features = self.process(lower_scale)
        upsampled = torch.nn.functional.interpolate(lower_features, scale_factor=2, mode="bilinear",
                                                    align_corners=False)
        H = data.shape[2]
        W = data.shape[3]

        output = self.post(torch.cat((features, upsampled), dim=1))

        return output


class SpatialFusionNet(torch.nn.Module):
    """ 
    The spatial fusion network generates the aleatoric uncertainty of an input depth map.
    """
    
    def __init__(self):

        super().__init__()

        self.activation = torch.nn.ReLU()

        cin = 4
        c0 = 24 
        cout = 1
        nlayers = 4 

        self.pre = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(cin, c0, kernel_size=3, stride=1, padding=0),
            torch.nn.ReLU(),
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(c0, c0, kernel_size=3, stride=1, padding=0),
            torch.nn.ReLU(),
        )

        self.process = UNet(c0, 2 * c0, 2 * c0, nlayers - 1, post_activation=nn.Identity())

        self.uncertainty_depth = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(3 * c0, 2 * c0, kernel_size=3, stride=1, padding=0),
            torch.nn.ReLU(),
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(2 * c0, c0, kernel_size=3, stride=1, padding=0),
            torch.nn.ReLU(),
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(c0, cout, kernel_size=3, stride=1, padding=0),
            torch.nn.ReLU()
        )

        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, data):

        features = self.pre(data)
        lower_scale = self.maxpool(features)
        lower_features = self.process(lower_scale)
        upsampled = torch.nn.functional.interpolate(lower_features, scale_factor=2, mode="bilinear",
                                                    align_corners=False)
        H = data.shape[2]
        W = data.shape[3]
        upsampled = upsampled[:, :, :H, :W]
        
        confidence = self.activation(self.uncertainty_depth(torch.cat((features, upsampled), dim=1)))

        return confidence


class ResBlock(nn.Module):
    """
    Basic residual block for Conv2d.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_1x1conv=False):
        super(ResBlock, self).__init__()

        self.convs = nn.ModuleList()
        self.convs.append(nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.2), 
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2), 
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        ))
        if use_1x1conv:
            self.convs.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0),
                nn.ReLU(True)
            ))
        else:
            self.convs.append( nn.Identity() )

    def forward(self, x):
        return self.convs[0](x) + self.convs[1](x)

class TemporalFusionNet(torch.nn.Module):
    """ 
    The temporal fusion network generates a mask for the image regions that have changed 
    from the previous frame, under the assumption that these represent dynamic objects in the scene.
    """

    def __init__(self):

        super().__init__()

        self.activation = torch.nn.Sigmoid() 

        dfeats_size = 24
        cfeats_size = 24
        self.dfeats = nn.Sequential(
            ResBlock(1 + 1, 8, 5, 1, 2, use_1x1conv=True),
            ResBlock(8, 16, 3, 1, 1, use_1x1conv=True), 
            ResBlock(16, dfeats_size, 3, 1, 1, use_1x1conv=True), 
        )
        self.cfeats = nn.Sequential(
            ResBlock(3 + 3, 8, 5, 1, 2, use_1x1conv=True),
            ResBlock(8, 16, 3, 1, 1, use_1x1conv=True), 
            ResBlock(16, cfeats_size, 3, 1, 1, use_1x1conv=True), 
        )

        cin = dfeats_size + cfeats_size + 3 + 1 
        c0 = 24 
        cout = 1
        nlayers = 4 

        self.pre = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(cin, c0, kernel_size=3, stride=1, padding=0),
            torch.nn.GroupNorm(1, c0),
            torch.nn.ReLU(),
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(c0, c0, kernel_size=3, stride=1, padding=0),
            torch.nn.GroupNorm(1, c0),
            torch.nn.ReLU(),
        )

        self.process = UNet(c0, 2 * c0, 2 * c0, nlayers - 1)
        
        self.post = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(3 * c0, c0, kernel_size=3, stride=1, padding=0),
            torch.nn.GroupNorm(1, c0),
            torch.nn.ReLU(),
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(c0, cout, kernel_size=3, stride=1, padding=0),
        )

        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, obv_depth, obv_rgb, prior_depth, prior_rgb ):
        
        # Scale down obvserved and prior inputs by half for feature extraction
        depth_feats_low = self.dfeats( torch.nn.functional.interpolate( torch.cat( (prior_depth, obv_depth), 1 ), 
                                                                        scale_factor=0.5, 
                                                                        mode='bilinear', 
                                                                        align_corners=False))
        rgb_feats_low = self.cfeats( torch.nn.functional.interpolate( torch.cat( (prior_rgb, obv_rgb), 1 ),
                                                                      scale_factor=0.5,
                                                                      mode='bilinear',
                                                                      align_corners=False))
        depth_feats = torch.nn.functional.interpolate( depth_feats_low, scale_factor=2, mode='bilinear', align_corners=True)
        rgb_feats = torch.nn.functional.interpolate( rgb_feats_low, scale_factor=2, mode='bilinear', align_corners=True)
                                   
        data = torch.cat((depth_feats, rgb_feats, obv_depth, obv_rgb), 1)

        features = self.pre(data)
        lower_scale = self.maxpool(features)
        lower_features = self.process(lower_scale)
        upsampled = torch.nn.functional.interpolate(lower_features, scale_factor=2, mode="bilinear", align_corners=False)

        
        alpha = self.activation(self.post(torch.cat((features, upsampled), dim=1)))

        return alpha
