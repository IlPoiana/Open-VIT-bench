import timm
import sys
import torch
from cvit_utils import load_cprd, plot_prediction_batch

if (len(sys.argv) == 2) :
    cprd_path = sys.argv[1]

    cprd = load_cprd(cprd_path)
    plot_prediction_batch(cprd)

else :
    print('Usage: convert_pt_cvit <cprd_path>')
