import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cvit_utils import plot_tensor, plot_prediction_batch
from cvit_utils import store_cvit, store_cpic, store_cprd, load_cvit, load_cpic, load_cprd

print("Test Utils")

vit = load_cvit('../test_files/test_utils.cvit')
pic = load_cpic('../test_files/test_utils.cpic')
prd = load_cprd('../test_files/test_utils.cprd')



print("### vit")
print(f'   num_classes: {vit.num_classes}')
print(f'   global_pool: {vit.global_pool}')
print(f'   embed_dim: {vit.embed_dim}')
print(f'   depth: {len(vit.blocks)}')

print(f'   has_class_token: {vit.has_class_token}')
print(f'   num_reg_tokens: {vit.num_reg_tokens}')
print(f'   num_prefix_tokens: {vit.num_prefix_tokens}')
print(f'   no_embed_class: {vit.no_embed_class}')

use_pos_embed = False if vit.pos_embed == None else True
print(f'   use_pos_embed: {use_pos_embed}')
use_pre_norm = False if type(vit.norm_pre) == nn.Identity else True
print(f'   use_pre_norm: {use_pre_norm}')
use_fc_norm = False if type(vit.fc_norm) == nn.Identity else True
print(f'   use_fc_norm: {use_fc_norm}')
print(f'   dynamic_img_size: {vit.dynamic_img_size}')



print("### pic")
plot_tensor(pic)
print("### prd")
plot_prediction_batch(prd)



store_cvit(vit, '../test_files/test_utils.cvit')
store_cpic(pic, '../test_files/test_utils.cpic')
store_cprd(prd, '../test_files/test_utils.cprd')
