import sys
import numpy as np
import struct



if (len(sys.argv) == 3) :
    measure_path = sys.argv[1]
    out_path = sys.argv[2]

    tot_load_cvit_time = 0.0
    tot_load_cpic_time = 0.0
    tot_patch_embed_time = 0.0
    tot_position_embed_time = 0.0
    tot_attn_time = 0.0
    tot_mlp_time = 0.0
    tot_pool_time = 0.0
    tot_head_time = 0.0
    tot_store_cprd_time = 0.0

    tot_pictures = 0
    tot_batches = 0
    tot_blocks = 0



    file = open(measure_path, 'rt')
    line = file.readline()
    # discarding "header"

    line = file.readline()
    while line != '' :
        data =  line.split(';')

        tot_pictures += int(data[0])
        depth = int(data[1])
        tot_blocks += depth
        tot_batches += 1

        tot_load_cvit_time += float(data[2])
        tot_load_cpic_time += float(data[3])
        tot_patch_embed_time += float(data[4])
        tot_position_embed_time += float(data[5])

        for i in range(depth) :
            tot_attn_time += float(data[6 + (2*i)])
            tot_mlp_time += float(data[7 + (2*i)])

        tot_pool_time += float(data[6+(2*depth)]) # 6+(2*depth) is 8+2*(depth-1)
        tot_head_time += float(data[7+(2*depth)])

        tot_store_cprd_time += float(data[8+(2*depth)])

        line = file.readline()
    file.close()

    tot_foreward_time = tot_patch_embed_time + tot_position_embed_time + tot_attn_time
    tot_foreward_time += tot_mlp_time + tot_pool_time + tot_head_time
    tot_time = tot_foreward_time + tot_load_cvit_time + tot_load_cpic_time + tot_store_cprd_time

    out_file = open(out_path, 'at')

    out_file.write(f'Analysis of {measure_path}:\n')
    out_file.write(f'   Number of batches: {tot_batches}\n')
    out_file.write(f'   Number of pictures: {tot_pictures}\n')
    out_file.write(f'   Total time to analyze the dataset: {tot_time}\n')
    out_file.write(f'   Average model load time: {tot_load_cvit_time/tot_batches} s\n')
    out_file.write('\n')

    out_file.write(f'   Average batch load time: {tot_load_cpic_time/tot_batches} s\n')
    out_file.write(f'   Average foreward time per batch: {tot_foreward_time/tot_batches} s\n')
    out_file.write(f'      Patch Embed: {tot_patch_embed_time/tot_batches} s\n')
    out_file.write(f'      Position Embed: {tot_position_embed_time/tot_batches} s\n')
    out_file.write(f'      Attention: {tot_attn_time/(tot_batches*tot_blocks)} s/block\n')
    out_file.write(f'      MLP: {tot_mlp_time/(tot_batches*tot_blocks)} s/block\n')
    out_file.write(f'      Pool: {tot_pool_time/tot_batches} s\n')
    out_file.write(f'      Head: {tot_head_time/tot_batches} s\n')
    out_file.write(f'   Average store time per prediction batch: {tot_store_cprd_time/tot_batches} s\n')
    out_file.write('\n')

    out_file.write(f'   Average picture load time: {tot_load_cpic_time/tot_pictures} s\n')
    out_file.write(f'   Average foreward time per picture: {tot_foreward_time/tot_pictures} s\n')
    out_file.write(f'      Patch Embed: {tot_patch_embed_time/tot_pictures} s\n')
    out_file.write(f'      Position Embed: {tot_position_embed_time/tot_pictures} s\n')
    out_file.write(f'      Attention: {tot_attn_time/(tot_pictures*tot_blocks)} s/block\n')
    out_file.write(f'      MLP: {tot_mlp_time/(tot_pictures*tot_blocks)} s/block\n')
    out_file.write(f'      Pool: {tot_pool_time/tot_pictures} s\n')
    out_file.write(f'      Head: {tot_head_time/tot_pictures} s\n')
    out_file.write(f'   Average store time per single prediction : {tot_store_cprd_time/tot_pictures} s\n')
    out_file.write('\n')

    out_file.close()

else :
    print('Usage: analyze_time_measuraments <measure_path> <out_path>')
