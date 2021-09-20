import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from protonets.models import register_model

from .utils import euclidean_dist



class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class Protonet(nn.Module):
    def __init__(self, encoder):
        super(Protonet, self).__init__()
        
        self.encoder = encoder

    def loss(self, sample):
        if type(sample)==type({}):
            xs=sample['xs']
            xq=sample['xq']
            classes=sample['class']
        else:
            xs=sample.xs
            xq=sample.xq 
            classes=sample.class_names
            xs_idxs=sample.xs_idxs          
            xq_idxs=sample.xq_idxs          

        n_class = xs.size(0)
        assert xq.size(0) == n_class
        n_support = xs.size(1)
        n_query = xq.size(1)

        target_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_query, 1).long()
        target_inds = Variable(target_inds, requires_grad=False)

        if xq.is_cuda:
            target_inds = target_inds.cuda()

        x = torch.cat([xs.view(n_class * n_support, *xs.size()[2:]),
                       xq.view(n_class * n_query, *xq.size()[2:])], 0)

        z = self.encoder.forward(x)
        z_dim = z.size(-1)

        z_proto = z[:n_class*n_support].view(n_class, n_support, z_dim).mean(1)
        zq = z[n_class*n_support:]

        #print(f"\n n_class: {n_class}, n_support: {n_support}, n_query: {n_query}")

        dists = euclidean_dist(zq, z_proto)
        #print(dists.size())
        log_p_y = F.log_softmax(-dists, dim=1).view(n_class, n_query, -1)

        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()

        _, y_hat = log_p_y.max(2)
        acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()

        if loss_val.isnan():
            print()
            print(loss_val.item())

            # print(classes)
            # print('xs:', xs_idxs)
            # print('xq:', xq_idxs)
            print(xs.shape, xq.shape)

            for i, (c,s,q) in enumerate(zip(classes,xs_idxs,xq_idxs)):
                print(i, c, s, q)

            #print()
            #print(zq)
            sys.exit("oooops: nan loss !!!!!")

        return loss_val, {
            'loss': loss_val.item(),
            'acc': acc_val.item()
        }

@register_model('protonet_conv')
def load_protonet_conv(**kwargs):
    x_dim = kwargs['x_dim']
    hid_dim = kwargs['hid_dim']
    z_dim = kwargs['z_dim']

    # print("x_dim: ", x_dim)
    # print("hid_dim: ", hid_dim)
    # print("z_dim: ", z_dim)

    def conv_block(in_channels, out_channels):
        # print("in_channels: ", in_channels)
        # print("out_channels: ", out_channels)
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            #nn.GELU(),
            nn.MaxPool2d(2)
        )

    encoder = nn.Sequential(
        conv_block(x_dim[0], hid_dim),# name='conv_block1'),
        conv_block(hid_dim, hid_dim),# name='conv_block2'),
        conv_block(hid_dim, hid_dim),# name='conv_block3'),
        conv_block(hid_dim, z_dim),# name='conv_block4'),
        Flatten()
    )

    return Protonet(encoder)


