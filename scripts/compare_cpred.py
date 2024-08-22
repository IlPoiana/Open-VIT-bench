import sys
import numpy as np
import struct

LABEL_UNSIGNED = 'I'
LABEL_FLOAT = 'f'
SIZEOF_UNSIGNED = 4
SIZEOF_FLOAT = 4



def load_cprd(path : str) :
    file = open(path, 'rb')
    ba = bytearray( file.read() )
    file.close()

    cvit_label = ba[0:4]
    ba = ba[4 : ]
    assert cvit_label == b'CPRD'

    b = struct.unpack(LABEL_UNSIGNED, ba[0:SIZEOF_UNSIGNED]) [0]
    cls = struct.unpack(LABEL_UNSIGNED, ba[SIZEOF_UNSIGNED:2*SIZEOF_UNSIGNED]) [0]
    ba = ba[2*SIZEOF_UNSIGNED : ]

    classes = []
    for i in range (b) :
        el = struct.unpack(LABEL_UNSIGNED, ba[i*SIZEOF_UNSIGNED : (i+1)*SIZEOF_UNSIGNED]) [0]
        classes.append(el)
    ba = ba[SIZEOF_UNSIGNED*b : ]

    prob = []
    for i in range (b) :
        el = struct.unpack(LABEL_FLOAT, ba[i*SIZEOF_FLOAT : (i+1)*SIZEOF_FLOAT]) [0]
        prob.append(el)
    ba = ba[SIZEOF_FLOAT*b : ]

    prob_matrix = []
    for i in range (b*cls) :
        el = struct.unpack(LABEL_FLOAT, ba[i*SIZEOF_FLOAT : (i+1)*SIZEOF_FLOAT]) [0]
        prob_matrix.append(el)
    ba = ba[SIZEOF_FLOAT*b*cls : ]

    prob_matrix = np.reshape(prob_matrix, (b, cls))

    return classes, prob, prob_matrix



if (len(sys.argv) == 6) :
    path_1 = sys.argv[1]
    path_2 = sys.argv[2]
    out_path = sys.argv[3]

    high_treshold = float(sys.argv[4])
    low_treshold = float(sys.argv[5])

    classes_1, prob_1, prob_matrix_1 = load_cprd(path_1)
    classes_2, prob_2, prob_matrix_2 = load_cprd(path_2)

    if (len(classes_1) == len(classes_2) and len(prob_1) == len(prob_2) and 
        len(prob_matrix_1) == len(prob_matrix_2) and len(prob_matrix_1[0]) == len(prob_matrix_2[0])) :

        out_file = open(out_path, 'at')
        out_file.write(f'Comparing {path_1} and {path_2}:\n')

        equal_classes = 0
        for i in range( len(classes_1) ) :
            if ( classes_1[i] == classes_2[i] ) :
                equal_classes += 1

        equal_class_probs = 0
        for i in range( len(prob_1) ) :
            diff = prob_1[i] - prob_2[i]
            if ( diff <= high_treshold and diff >= -high_treshold) :
                equal_class_probs += 1

        equal_probs = 0
        identical_probs = 0
        cumulative_relative_error = 0.0
        for i in range( len(prob_matrix_1) ) :
            for j in range( len(prob_matrix_1[0]) ) :
                diff = prob_matrix_1[i, j] - prob_matrix_2[i, j]
                relative_error = abs(diff) / abs(prob_matrix_1[i, j])
                cumulative_relative_error += relative_error
                if ( diff <= low_treshold and diff >= -low_treshold ) :
                    identical_probs += 1
                    equal_probs += 1
                elif ( diff <= high_treshold and diff >= -high_treshold ) :
                    equal_probs += 1
        mean_relative_error = cumulative_relative_error / (len(prob_matrix_1)*len(prob_matrix_1[0]))

        out_file.write(f'Equal classes:\n')
        out_file.write(f'   {equal_classes} out of {len(classes_1)}\n')
        out_file.write(f'Equal probabilities of selected class (difference < {high_treshold}):\n')
        out_file.write(f'   {equal_class_probs} out of {len(prob_1)}\n')
        out_file.write(f'Equal probabilities (difference < {high_treshold}):\n')
        out_file.write(f'   {equal_probs} out of {len(prob_matrix_1)*len(prob_matrix_1[0])}\n')
        out_file.write(f'Identical probabilities (difference < {low_treshold}):\n')
        out_file.write(f'   {identical_probs} out of {len(prob_matrix_1)*len(prob_matrix_1[0])}\n')
        out_file.write(f'Mean relative error:\n')
        out_file.write(f'   {mean_relative_error}\n')
        out_file.write('\n')
        out_file.close()

    else :
        print('The predictions are not comparable:')
        print(f'{path_1}: {len(classes_1)}, {len(prob_1)}, {len(prob_matrix_1)}x{len(prob_matrix_1[0])}')
        print(f'{path_2}: {len(classes_2)}, {len(prob_2)}, {len(prob_matrix_2)}x{len(prob_matrix_2[0])}')
        print('')

else :
    print('Usage: compare_cpred <cprd_1_path> <cprd_2_path> <out_path> <high_treshold> <low_treshold>')
