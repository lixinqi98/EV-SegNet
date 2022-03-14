import torch.nn as nn
import torch.nn.functional as F

def conv(in_channels, out_channels, kernel_size, strides=1, dilation_rate=1, use_bias=False):
    '''
    Apply a 2D convolution over an input signal composed several input planes.
    filter: Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution)

    tf version
    def conv(filters, kernel_size, strides=1, dilation_rate=1, use_bias=False):
    return layers.Conv2D(filters, kernel_size, strides=strides, padding='same', use_bias=use_bias,
                         kernel_regularizer=regularizers.l2(l=0.0003), dilation_rate=dilation_rate)
    '''
    # TODO: kernel_regularizer? 
    # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    return nn.Conv2d(in_channels=in_channels,
                     out_channels=out_channels, 
                     kernel_size=kernel_size,
                     stride=strides,
                     padding='same',
                     bias=use_bias,
                     dilation=dilation_rate
                     )


def separableConv(filters, kernel_size, strides=1, dilation_rate=1, use_bias=False):
    '''
    tf version def separableConv(filters, kernel_size, strides=1, dilation_rate=1, use_bias=False):
    return layers.SeparableConv2D(filters, kernel_size, strides=strides, padding='same', use_bias=use_bias,
                                  depthwise_regularizer=regularizers.l2(l=0.0003),
                                  pointwise_regularizer=regularizers.l2(l=0.0003), dilation_rate=dilation_rate)
    '''
    # TODO how to implement this in pytorch? No existing function
    # ref https://gist.github.com/bdsaglam/84b1e1ba848381848ac0a308bfe0d84c
    pass
    
def upsampling(inputs, scale):
    return F.interpolate(inputs, 
                         size=(input.size()[1], input.size()[2]), 
                         mode='bilinear', 
                         align_corners=False)

def reshape_into(inputs, input_to_copy):
    return F.interpolate(inputs, 
                         size=(input_to_copy.get_shape()[1].value, input_to_copy.get_shape()[2].value), 
                         mode='bilinear', 
                         align_corners=False)

def transposeConv(in_channels, filters, kernel_size, strides=1, dilation_rate=1, use_bias=False):
    return nn.ConvTranspose2d(in_channels=in_channels, out_channels=filters, stride=strides,
                              padding='same', use_bias=use_bias, dilation=dilation_rate)
    return layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding='same', use_bias=use_bias,
                                  kernel_regularizer=regularizers.l2(l=0.0003), dilation_rate=dilation_rate)


# Depthwise convolution
def depthwiseConv(kernel_size, strides=1, depth_multiplier=1, dilation_rate=1, use_bias=False):
    # TODO implement this in torch
    # ref https://gist.github.com/bdsaglam/b16de6ae6662e7a783e06e58e2c5185a
    pass


def get_features(features, name):
    def hook(model, input, output):
        features[name] = output.detach()
    return hook