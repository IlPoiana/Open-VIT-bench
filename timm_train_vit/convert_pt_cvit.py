import timm
import sys
import torch
from cvit_utils import store_cvit

if (len(sys.argv) == 3) :
    pt_path = sys.argv[1]
    cvit_path = sys.argv[2]

    vit = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=100)
    vit.load_state_dict(torch.load(pt_path))
    store_cvit(vit, cvit_path)

else :
    print('Usage: convert_pt_cvit <pt_path> <cvit_path>')
