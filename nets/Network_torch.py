from numpy import size
import torch
import torch.nn as nn
import torch.nn.functional as F
import pretrainedmodels
from nets.Network import Segception_v2
from network_utils import *

class Segception(nn.Module):
    def __init__(self, num_classes, input_shape=(None, None, 3), weights='imagenet', **kwargs):
        super(Segception, self).__init__(**kwargs)
        
        # load pretrained Xception
        base_model = pretrainedmodels.__dict__['xception'](num_classes=1000, pretrained='imagenet')
        
        # TODO get the each layer output here

        # Decoder
        # TODO test if in_channels numer set correctly
        self.adap_encoder_1 = EncoderAdaption(in_channels=512, filters=256, kernel_size=3, dilation_rate=1)
        self.adap_encoder_2 = EncoderAdaption(in_channels=512, filters=256, kernel_size=3, dilation_rate=1)
        self.adap_encoder_3 = EncoderAdaption(in_channels=256, filters=256, kernel_size=3, dilation_rate=1)
        self.adap_encoder_4 = EncoderAdaption(in_channels=256, filters=128, kernel_size=3, dilation_rate=1)
        self.adap_encoder_5 = EncoderAdaption(in_channels=128, filters=64, kernel_size=3, dilation_rate=1)

        self.decoder_conv_1 = FeatureGeneration(in_channels=256, filters=256, kernel_size=3, dilation_rate=2, blocks=5)
        self.decoder_conv_2 = FeatureGeneration(in_channels=256, filters=128, kernel_size=3, dilation_rate=2, blocks=5)
        self.decoder_conv_3 = FeatureGeneration(in_channels=128, filters=64, kernel_size=5, dilation_rate=1, blocks=4)
        self.decoder_conv_4 = FeatureGeneration(in_channels=64, filters=64, kernel_size=3, dilation_rate=1, blocks=2)
        self.aspp = ASPP(in_channels=64, filters=64, kernel_size=3)

        self.conv_logits = conv(in_channels=64, out_channels=num_classes, kernel_size=1, strides=1, use_bias=True)

    def call(self, inputs, training=None, mask=None):

            outputs = self.model_output(inputs, training=training)
            # add activations to the ourputs of the model
            for i in range(len(outputs)):
                outputs[i] = nn.LeakyReLU(negative_slope=0.3)(outputs[i])

            x = self.adap_encoder_1(outputs[0], training=training)
            x = upsampling(x, scale=2)
            x += reshape_into(self.adap_encoder_2(outputs[1], training=training), x) #512
            x = self.decoder_conv_1(x, training=training) #256

            x = upsampling(x, scale=2)
            x += reshape_into(self.adap_encoder_3(outputs[2], training=training), x)#256
            x = self.decoder_conv_2(x, training=training) #256

            x = upsampling(x, scale=2)
            x += reshape_into(self.adap_encoder_4(outputs[3], training=training), x)#128
            x = self.decoder_conv_3(x, training=training) #128
            x_aspp = self.aspp(x, training=training, operation='sum') #128
            x += x_aspp

            x = upsampling(x, scale=2)
            x += reshape_into(self.adap_encoder_5(outputs[4], training=training), x)  # 64
            x = self.decoder_conv_4(x, training=training)  # 64
            x = upsampling(x, scale=2)

            x = self.conv_logits(x)

            return x

class Segception_v2(nn.Model):
    def __init__(self, num_classes, input_shape=(None, None, 3), weights='imagenet', **kwargs):
        super(Segception_v2, self).__init__(**kwargs)
        # load pretrained Xception
        base_model = pretrainedmodels.__dict__['xception'](num_classes=1000, pretrained='imagenet')
        
        # TODO get the each layer output here

        # Decoder
        self.adap_encoder_1 = EncoderAdaption(in_channels=256, filters=256, kernel_size=3, dilation_rate=1)
        self.adap_encoder_2 = EncoderAdaption(in_channels, filters=256, kernel_size=3, dilation_rate=1)
        self.adap_encoder_3 = EncoderAdaption(in_channels, filters=256, kernel_size=3, dilation_rate=1)
        self.adap_encoder_4 = EncoderAdaption(in_channels, filters=128, kernel_size=3, dilation_rate=1)
        self.adap_encoder_5 = EncoderAdaption(in_channels, filters=64, kernel_size=3, dilation_rate=1)

        self.decoder_conv_1 = FeatureGeneration(in_channels, filters=256, kernel_size=3, dilation_rate=1, blocks=5)
        self.decoder_conv_2 = FeatureGeneration(in_channels, filters=128, kernel_size=3, dilation_rate=1, blocks=5)
        self.decoder_conv_3 = FeatureGeneration(in_channels, filters=64, kernel_size=3, dilation_rate=1, blocks=5)
        self.decoder_conv_4 = FeatureGeneration(in_channels, filters=64, kernel_size=3, dilation_rate=1, blocks=2)
        self.aspp = ASPP_2(in_channels, filters=64, kernel_size=3)

        self.conv_logits = conv(in_channels=in_channels, out_channels=num_classes, kernel_size=1, strides=1, use_bias=True)
        self.conv_logits_aux = conv(in_channels=in_channels, out_channels=num_classes, kernel_size=1, strides=1, use_bias=True)

    def call(self, inputs, training=None, mask=None, aux_loss=False):

        outputs = self.model_output(inputs, training=training)
        # add activations to the ourputs of the model
        for i in range(len(outputs)):
            outputs[i] = nn.LeakyReLU(negative_slope=0.3)(outputs[i])

        x = self.adap_encoder_1(outputs[0], training=training)
        x = upsampling(x, scale=2)
        x += reshape_into(self.adap_encoder_2(outputs[1], training=training), x)  # 512
        x = self.decoder_conv_1(x, training=training)  # 256

        x = upsampling(x, scale=2)
        x += reshape_into(self.adap_encoder_3(outputs[2], training=training), x)  # 256
        x = self.decoder_conv_2(x, training=training)  # 256

        x = upsampling(x, scale=2)
        x += reshape_into(self.adap_encoder_4(outputs[3], training=training), x)  # 128
        x = self.decoder_conv_3(x, training=training)  # 128

        x = self.aspp(x, training=training, operation='sum')  # 128
        x_aux = self.conv_logits_aux(x)
        x_aux = upsampling(x_aux, scale=2)
        x_aux_out = upsampling(x_aux, scale=2)

        x = upsampling(x, scale=2)
        x += reshape_into(self.adap_encoder_5(outputs[4], training=training), x)  # 64
        x = self.decoder_conv_4(torch.cat((x, x_aux), -1), training=training)  # 64
        x = self.conv_logits(x)
        x = upsampling(x, scale=2)

        if aux_loss:
            return x, x_aux_out
        else:
            return x

class Segception_v4(tf.keras.Model):
    def __init__(self, num_classes, input_shape=(None, None, 3), in_channels = 3, weights='imagenet', **kwargs):
        super(Segception_v4, self).__init__(**kwargs)
        # load pretrained Xception
        base_model = pretrainedmodels.__dict__['xception'](num_classes=1000, pretrained='imagenet')
        
        # TODO get the each layer output here

        # Decoder
        self.adap_encoder_1 = EncoderAdaption(in_channels, filters=256, kernel_size=3, dilation_rate=1)
        self.adap_encoder_2 = EncoderAdaption(in_channels, filters=256, kernel_size=3, dilation_rate=1)
        self.adap_encoder_3 = EncoderAdaption(in_channels, filters=256, kernel_size=3, dilation_rate=1)
        self.adap_encoder_4 = EncoderAdaption(in_channels, filters=128, kernel_size=3, dilation_rate=1)
        self.adap_encoder_5 = EncoderAdaption(in_channels, filters=64, kernel_size=3, dilation_rate=1)

        self.decoder_conv_1 = FeatureGeneration(in_channels, filters=256, kernel_size=3, dilation_rate=1, blocks=2)
        self.decoder_conv_2 = FeatureGeneration(in_channels, filters=128, kernel_size=3, dilation_rate=1, blocks=2)
        self.decoder_conv_3 = FeatureGeneration(in_channels, filters=64, kernel_size=3, dilation_rate=1, blocks=2)
        self.decoder_conv_4 = FeatureGeneration(in_channels, filters=32, kernel_size=3, dilation_rate=1, blocks=2)

        self.conv_logits = conv(in_channels=in_channels, out_channels=num_classes, kernel_size=1, strides=1, use_bias=True)

    def call(self, inputs, training=None, mask=None, aux_loss=False):

        outputs = self.model_output(inputs, training=training)
        # add activations to the ourputs of the model
        for i in range(len(outputs)):
            outputs[i] = nn.LeakyReLU(negative_slope=0.3)(outputs[i])

        x = self.adap_encoder_1(outputs[0], training=training)
        x = upsampling(x, scale=2)
        x += reshape_into(self.adap_encoder_2(outputs[1], training=training), x)  # 512
        x = self.decoder_conv_1(x, training=training)  # 256

        x = upsampling(x, scale=2)
        x += reshape_into(self.adap_encoder_3(outputs[2], training=training), x)  # 256
        x = self.decoder_conv_2(x, training=training)  # 256

        x = upsampling(x, scale=2)
        x += reshape_into(self.adap_encoder_4(outputs[3], training=training), x)  # 128
        x = self.decoder_conv_3(x, training=training)  # 128

        x = upsampling(x, scale=2)
        x += reshape_into(self.adap_encoder_5(outputs[4], training=training), x)  # 64
        x = self.decoder_conv_4(x, training=training)  # 64
        x = self.conv_logits(x)
        x = upsampling(x, scale=2)

        return x

class Segception_small(nn.Model):
    def __init__(self, num_classes, input_shape=(None, None, 3), in_channels=3, weights='imagenet', **kwargs):
        super(Segception_small, self).__init__(**kwargs)
        # load pretrained Xception
        base_model = pretrainedmodels.__dict__['xception'](num_classes=1000, pretrained='imagenet')
        
        # TODO get the each layer output here

        # Decoder
        self.adap_encoder_1 = EncoderAdaption(in_channels=3, filters=128, kernel_size=3, dilation_rate=1)
        self.adap_encoder_2 = EncoderAdaption(in_channels=3, filters=128, kernel_size=3, dilation_rate=1)
        self.adap_encoder_3 = EncoderAdaption(in_channels=3, filters=128, kernel_size=3, dilation_rate=1)
        self.adap_encoder_4 = EncoderAdaption(in_channels=3, filters=64, kernel_size=3, dilation_rate=1)
        self.adap_encoder_5 = EncoderAdaption(in_channels=3, filters=32, kernel_size=3, dilation_rate=1)

        self.decoder_conv_1 = FeatureGeneration(in_channels=3, filters=128, kernel_size=3, dilation_rate=1, blocks=3)
        self.decoder_conv_2 = FeatureGeneration(in_channels=3, filters=64, kernel_size=3, dilation_rate=1, blocks=3)
        self.decoder_conv_3 = FeatureGeneration(in_channels=3, filters=32, kernel_size=3, dilation_rate=1, blocks=3)
        self.decoder_conv_4 = FeatureGeneration(in_channels=3, filters=32, kernel_size=3, dilation_rate=1, blocks=1)
        self.aspp = ASPP_2(in_channels=3, filters=32, kernel_size=3)

        self.conv_logits = conv(in_channels=in_channels, out_channels=num_classes, kernel_size=1, strides=1, use_bias=True)

    def call(self, inputs, training=None, mask=None, aux_loss=False):

        outputs = self.model_output(inputs, training=training)
        # add activations to the ourputs of the model
        for i in range(len(outputs)):
            outputs[i] = nn.LeakyReLU(negative_slope=0.3)(outputs[i])

        x = self.adap_encoder_1(outputs[0], training=training)
        x = upsampling(x, scale=2)
        x += reshape_into(self.adap_encoder_2(outputs[1], training=training), x)  # 512
        x = self.decoder_conv_1(x, training=training)  # 256

        x = upsampling(x, scale=2)
        x += reshape_into(self.adap_encoder_3(outputs[2], training=training), x)  # 256
        x = self.decoder_conv_2(x, training=training)  # 256

        x = upsampling(x, scale=2)
        x += reshape_into(self.adap_encoder_4(outputs[3], training=training), x)  # 128
        x = self.decoder_conv_3(x, training=training)  # 128

        x = self.aspp(x, training=training, operation='sum')  # 128

        x = upsampling(x, scale=2)
        x += reshape_into(self.adap_encoder_5(outputs[4], training=training), x)  # 64
        x = self.decoder_conv_4(x, training=training)  # 64
        x = self.conv_logits(x)
        x = upsampling(x, scale=2)

        if aux_loss:
            return x, x
        else:
            return x

class Dilated_net(tf.keras.Model):
    def __init__(self, num_classes, input_shape=(None, None, 3), in_channels=3, weights='imagenet', **kwargs):
        super(Dilated_net, self).__init__(**kwargs)
        base_filter = 64
        self.conv1 = Conv_BN(in_channels=3, filters=base_filter, kernel_size=3, strides=2)
        self.encoder_conv_1 = FeatureGeneration(in_channels=3, filters=base_filter, kernel_size=3, dilation_rate=1, blocks=2)
        self.downsample_1 = DepthwiseConv_BN(in_channels=3, filters=base_filter, kernel_size=3, strides=2)
        self.encoder_conv_2 = FeatureGeneration(in_channels=3, filters=base_filter*2, kernel_size=3, dilation_rate=1, blocks=4)
        self.downsample_2 = DepthwiseConv_BN(in_channels=3, filters=base_filter*2, kernel_size=3, strides=2)
        self.encoder_conv_3 = FeatureGeneration(in_channels=3, filters=base_filter*4, kernel_size=3, dilation_rate=1, blocks=5)
        self.encoder_conv_4 = FeatureGeneration(in_channels=3, filters=base_filter*4, kernel_size=3, dilation_rate=2, blocks=4)
        self.encoder_conv_5 = FeatureGeneration(in_channels=3, filters=base_filter*4, kernel_size=3, dilation_rate=4, blocks=3)
        self.encoder_conv_6 = FeatureGeneration(in_channels=3, filters=base_filter*4, kernel_size=3, dilation_rate=8, blocks=2)
        self.encoder_conv_7 = FeatureGeneration(in_channels=3, filters=base_filter*4, kernel_size=3, dilation_rate=16, blocks=1)

        self.adap_encoder_1 = EncoderAdaption(in_channels=3, filters=base_filter*2, kernel_size=3, dilation_rate=1)


        #DepthwiseConv_BN
        self.decoder_conv_1 = FeatureGeneration(in_channels=3, filters=base_filter*2, kernel_size=3, dilation_rate=1, blocks=6)
        self.decoder_conv_2 = FeatureGeneration(in_channels=3, filters=base_filter, kernel_size=3, dilation_rate=1, blocks=3)
        self.aspp = ASPP_2(in_channels=3, filters=base_filter*2, kernel_size=3)

        self.conv_logits = conv(in_channels=in_channels, out_channels=num_classes, kernel_size=1, strides=1, use_bias=True)
        self.conv_logits_aux = conv(in_channels=in_channels, out_channels=num_classes, kernel_size=1, strides=1, use_bias=True)

    def call(self, inputs, training=None, mask=None, aux_loss=False):

        x = self.conv1(inputs, training=training)
        x = self.encoder_conv_1(x, training=training)
        x_enc = self.downsample_1(x, training=training)
        x = self.encoder_conv_2(x_enc, training=training)
        x = self.downsample_2(x, training=training)
        x1 = self.encoder_conv_3(x, training=training)
        x = x1 + self.encoder_conv_4(x1, training=training)
        x += self.encoder_conv_5(x + x1, training=training)
        x += self.encoder_conv_6(x + x1, training=training)
        x += self.encoder_conv_7(x + x1, training=training)
        x = upsampling(x + x1, scale=2)
        x = self.decoder_conv_1(x, training=training)

        x += self.adap_encoder_1(x_enc, training=training)

        x = self.aspp(x, training=training, operation='sum')  # 128
        x_aux = self.conv_logits_aux(x)
        x_aux = upsampling(x_aux, scale=2)
        x_aux_out = upsampling(x_aux, scale=2)

        x = upsampling(x, scale=2)
        x = self.decoder_conv_2(torch.cat((x, x_aux), -1), training=training)  # 64
        x = self.conv_logits(torch.cat((x, x_aux), -1))
        x = upsampling(x, scale=2)

        if aux_loss:
            return x, x_aux_out
        else:
            return x

class RefineNet(nn.Model):
    def __init__(self, num_classes, base_model, **kwargs):
        super(RefineNet, self).__init__(**kwargs)
        # mirar si cabe en memoria RefineNet
        # usar esta pero con restore self.model_output.variables y el call es con rgb y segmentacion
        # poner que justo esas a no optimizar, solo las nuevas
        self.base_model = base_model

        # Decoder
        self.conv = FeatureGeneration(in_channels=3, filters=32, kernel_size=3, dilation_rate=1, blocks=1)
        self.conv_logits = conv(filters=num_classes, kernel_size=1, strides=1, use_bias=True)

    def call(self, inputs, training=None, mask=None, iterations=1):
        # get non refined segmentation
        segmentation = self.base_model(inputs, training=training)

        for i in range(iterations):
            x = torch.cat([inputs, segmentation], -1)
            x = self.conv(x, training=training)
            segmentation = self.conv_logits(x)

        return segmentation

class Conv_BN(nn.Model):
    def __init__(self, in_channels, filters, kernel_size, strides=1) -> None:
        super(Conv_BN, self).__init__()

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides

        self.conv = conv(in_channels=in_channels, out_channels=filters, kernel_size=kernel_size, strides=strides)
        self.bn = nn.BatchNorm2d(eps=1e-3, momentum=0.993)
    
    def call(self, inputs, training=None, activation=True):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        if activation:
            x = nn.LeakyReLU(negative_slope=0.3)(x)
        return x


class DepthwiseConv_BN(nn.Model):
    def __init__(self, in_channels, filters, kernel_size, strides=1, dilation_rate=1):
        super(DepthwiseConv_BN, self).__init__()

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides

        self.conv = separableConv(filters=filters, kernel_size=kernel_size, strides=strides,
                                  dilation_rate=dilation_rate)
        self.bn = nn.BatchNorm2d(eps=1e-3, momentum=0.993)

    def call(self, inputs, training=None):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        x = nn.LeakyReLU(negative_slope=0.3)(x)

        return x


class Transpose_Conv_BN(nn.Model):
    def __init__(self, in_channels, filters, kernel_size, strides=1):
        super(Transpose_Conv_BN, self).__init__()

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides

        self.conv = transposeConv(filters=filters, kernel_size=kernel_size, strides=strides)
        self.bn = nn.BatchNorm2d(eps=1e-3, momentum=0.993)

    def call(self, inputs, training=None):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        x = nn.LeakyReLU(negative_slope=0.3)(x)

        return x

class ShatheBlock(nn.Model):
    def __init__(self, in_channels, filters, kernel_size,  dilation_rate=1, bottleneck=2) -> None:
        super(ShatheBlock, self).__init__()

        self.filters = filters * bottleneck
        self.kernel_size = kernel_size

        self.conv = DepthwiseConv_BN(in_channels, self.filters, kernel_size=kernel_size, dilation_rate=dilation_rate)
        self.conv1 = DepthwiseConv_BN(in_channels, self.filters, kernel_size=kernel_size, dilation_rate=dilation_rate)
        self.conv2 = DepthwiseConv_BN(in_channels, self.filters, kernel_size=kernel_size, dilation_rate=dilation_rate)
        self.conv3 = Conv_BN(in_channels, filters, kernel_size=1)
    
    def call(self, inputs, training=None):
        x = self.conv(inputs, training=training)
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        return x + inputs


class ShatheBlock_MultiDil(tf.keras.Model):
    def __init__(self, filters, kernel_size, dilation_rate=1, bottleneck=2):
        super(ShatheBlock_MultiDil, self).__init__()

        self.filters = filters * bottleneck
        self.filters_dil = filters / 2
        self.kernel_size = kernel_size

        self.conv = DepthwiseConv_BN(self.filters, kernel_size=kernel_size, dilation_rate=dilation_rate)
        self.conv1 = DepthwiseConv_BN(self.filters_dil, kernel_size=kernel_size, dilation_rate=dilation_rate*8)
        self.conv2 = DepthwiseConv_BN(self.filters_dil, kernel_size=kernel_size, dilation_rate=dilation_rate*4)
        self.conv3 = DepthwiseConv_BN(self.filters_dil, kernel_size=kernel_size, dilation_rate=dilation_rate*6)
        self.conv4 = DepthwiseConv_BN(self.filters_dil, kernel_size=kernel_size, dilation_rate=dilation_rate*2)
        self.conv5 = DepthwiseConv_BN(self.filters, kernel_size=kernel_size, dilation_rate=dilation_rate)
        self.conv6 = Conv_BN(filters, kernel_size=1)

    def call(self, inputs, training=None):
        x1 = self.conv(inputs, training=training)
        x2 = self.conv1(x1, training=training)
        x3 = self.conv2(x1, training=training)
        x4 = self.conv3(x1, training=training)
        x5 = self.conv4(x1, training=training)
        x6 = self.conv5(tf.concat((x2,x3,x4,x5), -1) + x1, training=training)
        x7 = self.conv6(x6, training=training)
        return x7 + inputs

class ASPP(nn.Model):
    def __init__(self, in_channels, filters, kernel_size):
        super(ASPP, self).__init__()

        self.conv1 = DepthwiseConv_BN(in_channels, filters, kernel_size=1, dilation_rate=1)
        self.conv2 = DepthwiseConv_BN(in_channels, filters, kernel_size=kernel_size, dilation_rate=4)
        self.conv3 = DepthwiseConv_BN(in_channels, filters, kernel_size=kernel_size, dilation_rate=8)
        self.conv4 = DepthwiseConv_BN(in_channels, filters, kernel_size=kernel_size, dilation_rate=16)
        self.conv5 = Conv_BN(in_channels, filters, kernel_size=1)

    def call(self, inputs, training=None, operation='concat'):
        # TODO: check if the inputs are tensor?
        feature_map_size = inputs.size()
        image_features = torch.mean(inputs, [1, 2], keep_dims=True)
        image_features = self.conv1(image_features, training=training)
        image_features = F.interpolate(image_features, size=(feature_map_size[1], feature_map_size[2]), mode='bilinear')
        x1 = self.conv2(inputs, training=training)
        x2 = self.conv3(inputs, training=training)
        x3 = self.conv4(inputs, training=training)
        if 'concat' in operation:
            x = self.conv5(torch.cat((image_features, x1, x2, x3, inputs), axis=3), training=training)
        else:
            x = image_features + x1 + x2 + x3 + inputs

        return x


class ASPP_2(tf.keras.Model):
    def __init__(self, in_channels, filters, kernel_size):
        super(ASPP_2, self).__init__()

        self.conv1 = DepthwiseConv_BN(in_channels, filters, kernel_size=1, dilation_rate=1)
        self.conv2 = DepthwiseConv_BN(in_channels, filters, kernel_size=kernel_size, dilation_rate=4)
        self.conv3 = DepthwiseConv_BN(in_channels, filters, kernel_size=kernel_size, dilation_rate=8)
        self.conv4 = DepthwiseConv_BN(in_channels, filters, kernel_size=kernel_size, dilation_rate=16)
        self.conv6 = DepthwiseConv_BN(in_channels, filters, kernel_size=kernel_size, dilation_rate=(2, 8))
        self.conv7 = DepthwiseConv_BN(in_channels, filters, kernel_size=kernel_size, dilation_rate=(6, 3))
        self.conv8 = DepthwiseConv_BN(in_channels, filters, kernel_size=kernel_size, dilation_rate=(8, 2))
        self.conv9 = DepthwiseConv_BN(in_channels, filters, kernel_size=kernel_size, dilation_rate=(3, 6))
        self.conv5 = Conv_BN(in_channels, filters, kernel_size=1)

    def call(self, inputs, training=None, operation='concat'):
        feature_map_size = inputs.size()
        image_features = torch.mean(inputs, [1, 2], keep_dims=True)
        image_features = self.conv1(image_features, training=training)
        image_features = F.interpolate(image_features, size=(feature_map_size[1], feature_map_size[2]), mode='bilinear')
        x1 = self.conv2(inputs, training=training)
        x2 = self.conv3(inputs, training=training)
        x3 = self.conv4(inputs, training=training)
        x4 = self.conv6(inputs, training=training)
        x5 = self.conv7(inputs, training=training)
        x4 = self.conv8(inputs, training=training) + x4
        x5 = self.conv9(inputs, training=training) + x5
        if 'concat' in operation:
            x = self.conv5(torch.cat((image_features, x1, x2, x3,x4,x5, inputs), axis=3), training=training)
        else:
            x = self.conv5(image_features + x1 + x2 + x3+x5+x4, training=training) + inputs

        return x




class DPC(tf.keras.Model):
    def __init__(self, filters):
        super(DPC, self).__init__()

        self.conv1 = DepthwiseConv_BN(filters, kernel_size=3, dilation_rate=(1, 6))
        self.conv2 = DepthwiseConv_BN(filters, kernel_size=3, dilation_rate=(18, 15))
        self.conv3 = DepthwiseConv_BN(filters, kernel_size=3, dilation_rate=(6, 21))
        self.conv4 = DepthwiseConv_BN(filters, kernel_size=3, dilation_rate=(1, 1))
        self.conv5 = DepthwiseConv_BN(filters, kernel_size=3, dilation_rate=(6, 3))

    def call(self, inputs, training=None, operation='concat'):
        x1 = self.conv1(inputs, training=training)
        x2 = self.conv2(x1, training=training)
        x3 = self.conv3(x1, training=training)
        x4 = self.conv4(x1, training=training)
        x5 = self.conv5(x2, training=training)

        if 'concat' in operation:
            x = tf.concat((x1, x2, x3, x4, x5, inputs), axis=3)
        else:
            x = x1 + x2 + x3 + x4 + x5 + inputs
        return x


class EncoderAdaption(nn.Model):
    def __init__(self, in_channels, filters, kernel_size, dilation_rate=1) -> None:
        super(EncoderAdaption, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size

        self.conv1 = Conv_BN(in_channels, filters, kernel_size=1)
        self.conv2 = ShatheBlock(in_channels, filters, kernel_size=kernel_size, dilation_rate=dilation_rate)
    
    def call(self, inputs, training=None):
        x = self.conv1(inputs, training=training)
        x = self.conv2(x, training=training)
        return x


class FeatureGeneration(nn.Model):
    def __init__(self, in_channels, filters, kernel_size, dilation_rate=1, blocks=3):
        super(FeatureGeneration, self).__init__()

        self.filters = filters
        self.kernel_size = kernel_size

        self.conv0 = Conv_BN(in_channels, self.filters, kernel_size=1)
        self.blocks = []
        for n in range(blocks):
            self.blocks = self.blocks + [
                ShatheBlock(in_channels, self.filters, kernel_size=kernel_size, dilation_rate=dilation_rate)]

    def call(self, inputs, training=None):

        x = self.conv0(inputs, training=training)
        for block in self.blocks:
            x = block(x, training=training)

        return x

class FeatureGeneration_Dil(tf.keras.Model):
    def __init__(self, filters, kernel_size, dilation_rate=1,  blocks=3):
        super(FeatureGeneration_Dil, self).__init__()

        self.filters = filters
        self.kernel_size = kernel_size

        self.conv0 = Conv_BN(self.filters, kernel_size=1)
        self.blocks = []
        for n in xrange(blocks):
            self.blocks = self.blocks + [
                ShatheBlock_MultiDil(self.filters, kernel_size=kernel_size, dilation_rate=dilation_rate)]

    def call(self, inputs, training=None):

        x = self.conv0(inputs, training=training)
        for block in self.blocks:
            x = block(x, training=training)

        return x