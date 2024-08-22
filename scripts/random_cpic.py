import sys
import torch
import random
import struct

LABEL_UNSIGNED = 'I'
LABEL_FLOAT = 'f'

if (len(sys.argv) == 9) :
    path = sys.argv[1]

    min_B = int(sys.argv[2])
    max_B = int(sys.argv[3])
    B = random.randint(min_B, max_B)

    C = int(sys.argv[4])
    H = int(sys.argv[5])
    W = int(sys.argv[6])

    min_val = float(sys.argv[7])
    max_val = float(sys.argv[8])

    pic = torch.rand(B, C, H, W)

    ba = bytearray(b'CPIC')
    ba.extend( bytearray(struct.pack(LABEL_UNSIGNED, pic.dim())) )
    for sh in pic.shape :
        ba.extend( bytearray(struct.pack(LABEL_UNSIGNED, sh)) )

    data = torch.reshape(pic, (-1,)).tolist()
    for el in data :
        el = el * (max_val-min_val) + min_val
        ba.extend( bytearray(struct.pack(LABEL_FLOAT, el)) )

    file = open(path, 'wb')
    file.write(ba)
    file.close()

else :
    print('Usage: random_cpic <path> <min_B> <max_B> <C> <H> <W> <min_val> <max_val>')
