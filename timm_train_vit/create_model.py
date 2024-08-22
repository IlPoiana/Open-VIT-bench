import timm
import sys
import torch
from cvit_utils import store_cvit

if (len(sys.argv) == 5) :
    cvit_1_path = sys.argv[1]
    cvit_2_path = sys.argv[2]
    pt_1_path = sys.argv[3]
    pt_2_path = sys.argv[4]

    vit_1 = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=100)
    store_cvit(vit_1, cvit_1_path)
    torch.save(vit_1.state_dict(), pt_1_path)

    vit_2 = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=100)
    store_cvit(vit_2, cvit_2_path)
    torch.save(vit_2.state_dict(), pt_2_path)

else :
    print('Usage: create_model <cvit_1_path> <cvit_2_path> <pt_1_path> <pt_2_path>')
