import sys
import time
import timm
import torch

from cvit_utils import load_cvit, load_cpic, tensor_2d_to_prediction_batch, store_cprd

#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

if (len(sys.argv) == 5) :
    pt_path = sys.argv[1]
    cpic_path = sys.argv[2]
    cprd_path = sys.argv[3]
    measure_path = sys.argv[4]

    vit = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=100)
    start_time = time.perf_counter()

    #vit = load_cvit(cvit_path)
    device = torch.device('cpu')
    vit.load_state_dict(torch.load(pt_path, map_location=device))
    vit = vit.cpu()

    end_time = time.perf_counter()
    load_cvit_time = end_time - start_time

    start_time = time.perf_counter()
    pic = load_cpic(cpic_path)
    pic = pic.cpu()
    end_time = time.perf_counter()
    load_cpic_time = end_time - start_time

    pred, times = vit.timed_forward(pic)
    start_time = time.perf_counter()
    pred = tensor_2d_to_prediction_batch(pred)
    end_time = time.perf_counter()
    times[-1] += end_time - start_time

    start_time = time.perf_counter()
    store_cprd(pred, cprd_path)
    end_time = time.perf_counter()
    store_cprd_time = end_time - start_time

    measure_file = open(measure_path, 'at')
    depth = len(vit.blocks)
    measure_file.write(f'{pred.b};{depth};{load_cvit_time};{load_cpic_time};')
    for t in times :
        measure_file.write(f'{t};')
    measure_file.write(f'{store_cprd_time}\n')
    measure_file.close()

else :
    print('Usage: vit <pt_path> <cpic_path> <cprd_path> <measure_file_path>')
