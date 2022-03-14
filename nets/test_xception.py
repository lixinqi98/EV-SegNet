# %%
import torch
from xception import xception

features = {}
def get_features(name):
    def hook(model, input, output):
        features[name] = output.detach()
    return hook

# create the hook

base_model = xception(num_classes=1000, pretrained='imagenet')
print(base_model)

base_model.block2.rep[2].register_forward_hook(get_features('block2_sepconv2_bn'))
base_model.block3.rep[2].register_forward_hook(get_features('block3_sepconv2_bn'))
base_model.block4.rep[2].register_forward_hook(get_features('block4_sepconv2_bn'))
base_model.block11.rep[2].register_forward_hook(get_features('block11_sepconv2_bn'))
base_model.block12.rep[2].register_forward_hook(get_features('block12_sepconv2_bn'))

#%%
x = torch.randn(32, 3, 299, 299)
output = base_model(x)

# %%
