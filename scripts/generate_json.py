import pandas as pd

import sbatchman as sbm
import argparse
import json
import os

#path
JSON_PATH = "./json/"
BLOCK_PATH = "blocks/"
#flags
CONV2D = "--conv2d"
LAYERNORM = "--ln"
ATTENTION = "--att"
MLP = "--mlp"

#tags
ALL_LEVELS="all_levels_img"
HIGH_LEVEL="high_levels_batch"
STREAMS="streams"

parser = argparse.ArgumentParser(description="put the correspondent block")
parser.add_argument(CONV2D, action="store_true",help="convolutional block of the patch embedder") # expect an argument
parser.add_argument(LAYERNORM, action="store_true",help= "layer normalization block")
parser.add_argument(ATTENTION, action="store_true", help="attention block")


if __name__ == "__main__":

    # put the possibility of flags do all the runs of one config or just do a single one
    args = parser.parse_args()
    cluster_name = 'baldo'

    if(args.conv2d):
        print("conv2d")
        component_path = "conv2d/"
        config_name = "conv2d_test"
        

        for tag in [ALL_LEVELS, HIGH_LEVEL, STREAMS]:
            full_path = JSON_PATH + BLOCK_PATH + component_path + tag
            os.makedirs(full_path, exist_ok=True)

            # fetch the runs from sbatchman
            jobs = sbm.jobs_df(cluster_name=cluster_name, config_name=config_name, tag= tag)
            job_list = sbm.jobs_list(cluster_name, config_name=config_name, status= [sbm.Status.COMPLETED], tag=tag)
            # filter only for the COMPLETED jobs
            completed_jobs = jobs[jobs["status"] == 'COMPLETED']
            if completed_jobs.shape[0] != len(job_list):
                print("something broken with job fetching")
                exit()

            # for every run find the param setup of that run (in the name rn) and create a json file with the std. out
            command_completed = completed_jobs["command"]
            list_command_completed = []
            for idx,cmd in enumerate(command_completed):
                cmd_list = cmd.split(' ')
                # print(cmd_list)
                
                b_img = (cmd_list[1].split('/'))[1]
                batch, img = b_img.split('-')
                level = cmd_list[3]
                stream_n = cmd_list[4]
                if(tag == STREAMS):
                    file_name = batch + "-" + img + "-" + level + "-" + stream_n
                else:
                    file_name = batch + "-" + img + "-" + level
                list_command_completed.append(file_name)
                json_data = json.loads(job_list[idx].get_stdout())
                
                f = open(full_path + "/" + file_name + ".json", "w")
                json.dump(json_data, f, indent=4)
                f.close()
    
    if(args.ln):
        print("layer-norm")
        ##JSON GEN FOR ln
    if(args.att):
        print("attention")
        ##JSON GEN FOR attention

    
    # print(list_command_completed)

    # save the json file with the right format
