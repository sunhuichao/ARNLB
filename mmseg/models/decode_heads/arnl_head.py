# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.cnn import NonLocal2d
from torch import nn

from mmseg.registry import MODELS
from .fcn_head import FCNHead
##ARNLB
class NewNonLocal2d(NonLocal2d):
    def __init__(self, *arg, **kwargs):
        super().__init__(*arg, **kwargs)
        #self.temperature = temperature
        #self.w1 = torch.nn.Parameter(torch.rand(1), requires_grad=True).cuda()
        #self.sigma_x = nn.Conv2d(self.in_channels, out_channels=self.inter_channels,kernel_size=1)
        self.lambda_x = nn.Conv2d(self.in_channels, out_channels=1,kernel_size=3, stride=1, padding=1)

    def embedded_gaussian(self, theta_x, phi_x):
        # NonLocal1d pairwise_weight: [N, H, H]
        # NonLocal2d pairwise_weight: [N, HxW, HxW]
        # NonLocal3d pairwise_weight: [N, TxHxW, TxHxW]
        pairwise_weight = torch.matmul(theta_x, phi_x)
        if self.use_scale:
            # theta_x.shape[-1] is `self.inter_channels`
            #pairwise_weight /= theta_x.shape[-1]**0.5
            pairwise_weight /= 1
        pairwise_weight = pairwise_weight.softmax(dim=-1)
        return pairwise_weight

    def forward(self, x):
        # x: [N, C, H, W]
        n = x.size(0)
        g_x = self.g(x).view(n,self.inter_channels,-1)
        g_x = g_x.permute(0,2,1)


        # theta_x: [N, HxW, C], phi_x: [N, C, HxW]  Q
        if self.mode == 'gaussian':
            theta_x = x.view(n, self.in_channels, -1) #Q
            theta_x = theta_x.permute(0, 2, 1)
            lambda_x  = x.view(n,self.in_channels,-1)
            lambda_x = lambda_x.permute(0,2,1)
            if self.sub_sample:
                phi_x = self.phi(x).view(n, self.in_channels, -1)  # K
                #sigma_x = self.sigma_x.view(n,self.in_channels,-1)
            else:
                phi_x = x.view(n, self.in_channels, -1)
                #sigma_x = x.view(n,self.in_channels,-1)

        elif self.mode == 'concatenation':
            theta_x = self.theta(x).view(n, self.inter_channels, -1, 1) #Q
            lambda_x = self.lambda_x(x).view(n, self.inter_channels, -1, 1)
            #sigma_x = self.sigma_x(x).view(n, self.inter_channels, 1, -1)
            phi_x = self.phi(x).view(n, self.inter_channels, 1, -1)  # K

        else:
            theta_x = self.theta(x).view(n, self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            #print('theta_x',theta_x.size()) #theta_x torch.Size([2, 8192, 256])
            lambda_x = self.lambda_x(x).view(n,1,-1)
            #print('lambda_x1', lambda_x.size()) # lambda_x1 torch.Size([2, 1, 8192])
            lambda_x = lambda_x.permute(0, 2, 1) #lambda_x torch.Size([2, 8192, 1])
            #print('lambda_x', lambda_x.size())
            phi_x = self.phi(x).view(n, self.inter_channels, -1)  #phi_x torch.Size([2, 256, 8192])
            #print('phi_x', phi_x.size())
            phi_x = phi_x.permute(0, 2, 1)#sigma_x torch.Size([2, 256, 8192])
        #DotPr = torch.mul(h_x,theta_x)  # diancheng Dotpr torch.Size([2, 8192, 256])

        #print('Dotpr',DotPr.size())

        pairwise_func = getattr(self, self.mode)
        # pairwise_weight: [N, HxW, HxW]
        lambda_x = lambda_x + torch.zeros(self.inter_channels).cuda()
        #print('lambda_xguangbo',lambda_x.size()) #lambda_xguangbo torch.Size([2, 8192, 256])


        DotPr = torch.mul(lambda_x, theta_x) #DotPr torch.Size([2, 8192, 256])

        
        pairwise_weight = pairwise_func(DotPr, phi_x)
        # y: [N, HxW, C]
        #print('g_x', g_x.size()) #g_x torch.Size([2, 8192, 256])
        y = torch.matmul(pairwise_weight, g_x)  # jia quan gei value


        # y: [N, C, H, W]
        y = y.permute(0, 2, 1).contiguous().reshape(n, self.inter_channels,
                                    *x.size()[2:])
        #print('y', y.size())




        output = x + self.conv_out(y)

        return output


@MODELS.register_module()
class ARNLHead(FCNHead):
    def __init__(self,
                 reduction=2,
                 use_scale=False,
                 mode='embedded_gaussian',
                 temperature=0.05,
                 **kwargs):
        super(ARNLHead, self).__init__(num_convs=2, **kwargs)

        self.reduction = reduction
        self.use_scale = use_scale
        self.mode = mode
        self.temperature = temperature
        self.non_blocak = NewNonLocal2d(
            in_channels=self.channels,
            reduction=self.reduction,
            use_scale=self.use_scale,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            mode=self.mode,
            )

    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)
        #print(x.size()) torch.Size([2, 2048, 64, 128])
        output = self.convs[0](x)
        #print(output.size())
        output = self.non_blocak(output)
        output = self.convs[1](output)

        if self.concat_input:
            output = self.conv_cat(torch.cat([x, output], dim=1))
        output = self.cls_seg(output)

        return output

