import sys
import numpy as np
import struct



if (len(sys.argv) == 3) :
    cprd_path = sys.argv[1]
    out_path = sys.argv[2]

    equal_classes = 0
    equal_class_probs = 0
    equal_probs = 0
    identical_probs = 0

    tot_classes = 0
    tot_class_probs = 0
    tot_probs = 0

    cumulative_mre = 0.0
    tot_cprd = 0

    file = open(cprd_path, 'rt')
    line = file.readline()

    first_batch = True
    while line != '' :
        # discarding "comparing file and file\n"
        line = file.readline()
        # discarding "Equal classes:\n"
        line = file.readline()

        data = line.split(' out of ')
        equal_classes += int(data[0])
        tot_classes += int(data[1])

        line = file.readline()
        # discarding "Equal probabilities of selected class (difference < high_treshold):\n"
        if (first_batch) :
            high_treshold = line.split(' ')
            high_treshold = high_treshold[7]
            high_treshold = high_treshold[ : -3]
        line = file.readline()

        data =  line.split(' out of ')
        equal_class_probs += int(data[0])
        tot_class_probs += int(data[1])

        line = file.readline()
        # discarding "Equal probabilities (difference < high_treshold):"
        line = file.readline()

        data =  line.split(' out of ')
        equal_probs += int(data[0])
        tot_probs += int(data[1])

        line = file.readline()
        if (first_batch) :
            low_treshold = line.split(' ')
            low_treshold = low_treshold[4]
            low_treshold = low_treshold[ : -3]
            first_batch = False
        # discarding "Identical probabilities (difference < low_treshold):"
        line = file.readline()

        data =  line.split(' out of ')
        identical_probs += int(data[0])

        line = file.readline()
        # discarding "Mean relative error:"
        line = file.readline()

        cumulative_mre += float(line)
        tot_cprd += 1

        line = file.readline()
        # discarding trailer blank line
        line = file.readline()
    file.close()
    average_mre = cumulative_mre / tot_cprd

    out_file = open(out_path, 'at')

    out_file.write(f'Analysis of {cprd_path}:\n')
    out_file.write(f'Equal classes:\n')
    out_file.write(f'   {equal_classes} out of {tot_classes}\n')
    out_file.write(f'Equal probabilities of selected class (difference < {high_treshold}):\n')
    out_file.write(f'   {equal_class_probs} out of {tot_class_probs}\n')
    out_file.write(f'Equal probabilities (difference < {high_treshold}):\n')
    out_file.write(f'   {equal_probs} out of {tot_probs}\n')
    out_file.write(f'Identical probabilities (difference < {low_treshold}):\n')
    out_file.write(f'   {identical_probs} out of {tot_probs}\n')
    out_file.write(f'Average Mean Relative Error:\n')
    out_file.write(f'   {average_mre}\n')
    out_file.write('\n')
    out_file.close()

else :
    print('Usage: summary_cpred_comparison <cprd_path> <out_path>')
