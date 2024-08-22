import sys
import torch

min_val = -1.0
max_val = 1.0

if (len(sys.argv) == 1) :
    print('Please, enter the dimensions as command line arguments')

elif (len(sys.argv) == 2) :
    DIM = int(sys.argv[1])
    print(f'Creating RowVector[{DIM}]\n')
    x = torch.rand(DIM)

    print('x = torch.tensor([', end='')
    for i in range(DIM) :
        val = (x[i].item() * (max_val-min_val)) + min_val
        if (i == DIM-1) :
            print(f'{val:.3f}', end='])\n\n')
        else :
            print(f'{val:.3f}', end=', ')

    print(f'vit_float x_data[{DIM}]', end=' = {')
    for i in range(DIM) :
        val = (x[i].item() * (max_val-min_val)) + min_val
        if (i == DIM-1) :
            print(f'{val:.3f}', end='};\n')
        else :
            print(f'{val:.3f}', end=', ')
    print(f'RowVector x(x_data, {DIM});\n')

elif (len(sys.argv) == 3) :
    ROWS = int(sys.argv[1])
    COLS = int(sys.argv[2])
    print(f'Creating Matrix[{ROWS}x{COLS}]\n')
    x = torch.rand(ROWS, COLS)

    print('x = torch.tensor([', end='')
    for i in range(ROWS) :
        print('[', end='')
        for j in range(COLS) :
            val = (x[i, j].item() * (max_val-min_val)) + min_val
            if (j == COLS-1) :
                if (i == ROWS-1) :
                    print(f'{val:7.3f}', end=']])\n\n')
                else :
                    print(f'{val:7.3f}', end='],\n')
                    print('                  ', end='')
            else :
                print(f'{val:7.3f}', end=', ')

    print(f'vit_float x_data[{ROWS}*{COLS}]', end=' = {\n')
    for i in range(ROWS) :
        print('    ', end='')
        for j in range(COLS) :
            val = (x[i, j].item() * (max_val-min_val)) + min_val
            if (j == COLS-1) :
                if (i == ROWS-1) :
                    print(f'{val:7.3f}', end='\n};\n')
                else :
                    print(f'{val:7.3f}', end=',\n')
            else :
                print(f'{val:7.3f}', end=', ')
    print(f'Matrix x(x_data, {ROWS}*{COLS}, {ROWS}, {COLS});\n')

elif (len(sys.argv) == 4) :
    B = int(sys.argv[1])
    N = int(sys.argv[2])
    C = int(sys.argv[3])
    print(f'Creating Tensor[{B}x{N}x{C}]\n')
    x = torch.rand(B, N, C)

    print('x = torch.tensor([', end='')
    for b in range(B) :
        print('[', end='')
        for n in range(N) :
            print('[', end='')
            for c in range(C) :
                val = (x[b, n, c].item() * (max_val-min_val)) + min_val
                if (c == C-1) :
                    if (n == N-1) :
                        if (b == B-1) :
                            print(f'{val:7.3f}', end=']]])\n\n')
                        else :
                            print(f'{val:7.3f}', end=']],\n\n')
                            print('                  ', end='')
                    else :
                        print(f'{val:7.3f}', end='],\n')
                        print('                   ', end='')
                else :
                    print(f'{val:7.3f}', end=', ')

    print(f'vit_float x_data[{B}*{N}*{C}]', end=' = {\n')
    for b in range(B) :
        for n in range(N) :
            print('    ', end='')
            for c in range(C) :
                val = (x[b, n, c].item() * (max_val-min_val)) + min_val
                if (c == C-1) :
                    if (n == N-1) :
                        if (b == B-1) :
                            print(f'{val:7.3f}', end='\n};\n')
                        else :
                            print(f'{val:7.3f}', end=',\n\n')
                    else :
                        print(f'{val:7.3f}', end=',\n')
                else :
                    print(f'{val:7.3f}', end=', ')
    print(f'Tensor x(x_data, {B}*{N}*{C}, {B}, {N}, {C});\n')

elif (len(sys.argv) == 5) :
    B = int(sys.argv[1])
    C = int(sys.argv[2])
    H = int(sys.argv[3])
    W = int(sys.argv[4])
    print(f'Creating Picture[{B}x{C}x{H}x{W}]\n')
    x = torch.rand(B, C, H, W)

    print('x = torch.tensor([', end='')
    for b in range(B) :
        print('[', end='')
        for c in range(C) :
            print('[', end='')
            for h in range(H) :
                print('[', end='')
                for w in range(W) :
                    val = (x[b, c, h, w].item() * (max_val-min_val)) + min_val
                    if (w == W-1) :
                        if (h == H-1) :
                            if (c == C-1) :
                                if (b == B-1) :
                                    print(f'{val:7.3f}', end=']]]])\n\n')
                                else :
                                    print(f'{val:7.3f}', end=']]],\n\n\n\n')
                                    print('                  ', end='')
                            else :
                                print(f'{val:7.3f}', end=']],\n\n')
                                print('                   ', end='')
                        else :
                            print(f'{val:7.3f}', end='],\n')
                            print('                    ', end='')
                    else :
                        print (f'{val:7.3f}', end=', ')

    print(f'vit_float x_data[{B}*{C}*{H}*{W}]', end=' = {\n')
    for b in range(B) :
        for c in range(C) :
            for h in range(H) :
                print('    ', end='')
                for w in range(W) :
                    val = (x[b, c, h, w].item() * (max_val-min_val)) + min_val
                    if (w == W-1) :
                        if (h == H-1) :
                            if (c == C-1) :
                                if (b == B-1) :
                                    print(f'{val:7.3f}', end='\n};\n')
                                else :
                                    print(f'{val:7.3f}', end=',\n\n\n\n')
                            else :
                                print(f'{val:7.3f}', end=',\n\n')
                        else :
                            print(f'{val:7.3f}', end=',\n')
                    else :
                        print (f'{val:7.3f}', end=', ')
    print(f'PictureBatch x(x_data, {B}*{C}*{H}*{W}, {B}, {C}, {H}, {W});\n')

else :
    print('Too many arguments')
