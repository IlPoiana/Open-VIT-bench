import sys
import torch.nn as nn
from cvit_utils import load_cvit

if (len(sys.argv) == 3) :
    model_path = sys.argv[1]
    out_path = sys.argv[2]
    vit = load_cvit(model_path)

    out_file = open(out_path, 'at')
    out_file.write(f'{model_path} info:\n')
    out_file.write(f'   num_classes: {vit.num_classes}\n')
    out_file.write(f'   global_pool: {vit.global_pool}\n')
    out_file.write(f'   embed_dim: {vit.embed_dim}\n')
    out_file.write(f'   depth: {len(vit.blocks)}\n')

    out_file.write(f'   has_class_token: {vit.has_class_token}\n')
    out_file.write(f'   num_reg_tokens: {vit.num_reg_tokens}\n')
    out_file.write(f'   num_prefix_tokens: {vit.num_prefix_tokens}\n')
    out_file.write(f'   no_embed_class: {vit.no_embed_class}\n')

    use_pos_embed = False if vit.pos_embed == None else True
    out_file.write(f'   use_pos_embed: {use_pos_embed}\n')
    use_pre_norm = False if type(vit.norm_pre) == nn.Identity else True
    out_file.write(f'   use_pre_norm: {use_pre_norm}\n')
    use_fc_norm = False if type(vit.fc_norm) == nn.Identity else True
    out_file.write(f'   use_fc_norm: {use_fc_norm}\n')
    out_file.write(f'   dynamic_img_size: {vit.dynamic_img_size}\n')
    out_file.write('\n')
    out_file.close()

else :
    print('Usage: print_model_info <model_path> <out_path>')
