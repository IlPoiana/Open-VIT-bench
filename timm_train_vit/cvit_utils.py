import torch
import torch.nn as nn
import struct

from timm.models.vision_transformer import VisionTransformer, Attention, LayerScale, Block
from timm.layers.patch_embed import PatchEmbed
from timm.layers.mlp import Mlp



#
# Plot tensors
#

def plot_tensor(t : torch.Tensor) :
    if (t.dim() == 1) :
        DIM = t.shape[0]
        print(f'RowVector[{DIM}]:')
        print('   ', end='')
        for i in range(DIM) :
            print(f'{t[i].item():.3f}', end=' ')
        print('\n')

    elif (t.dim() == 2) :
        ROWS, COLS = t.shape
        print(f'Matrix[{ROWS}x{COLS}]:')
        for i in range(ROWS) :
            print('   ', end='')
            for j in range(COLS) :
                print(f'{t[i, j].item():7.3f}', end=' ')
            print('')
        print('')

    elif (t.dim() == 3) :
        B, N, C = t.shape
        print(f'Tensor[{B}x{N}x{C}]:')
        for b in range(B) :
            print(f'   B[{b}]')
            for n in range(N) :
                print('   ', end='')
                for c in range(C) :
                    print(f'{t[b, n, c].item():7.3f}', end=' ')
                print('')
        print('')

    elif (t.dim() == 4) :
        B, C, H, W = t.shape
        print(f'Picture[{B}x{C}x{H}x{W}]:')
        for i in range(B) :
            print(f'   B[{i}]')
            for j in range(H) :
                print('   ', end='')
                for k in range(W) :
                    print('[ ', end='')
                    for l in range(C) :
                        print(f'{t[i, l, j, k].item():7.3f}', end=' ')
                    print(']', end='')
                print('')
        print('')

    else :
        print('Tensor[', end='')
        for i in range(t.dim()) :
            if (i == t.dim()-1) :
                print(t.shape[i], end=']\n\n')
            else :
                print(t.shape[i], end='x')



#
# PredictionBatch class
#

class PredictionBatch :
    b : int
    cls : int

    classes : torch.Tensor
    prob : torch.Tensor
    prob_matrix : torch.Tensor

def tensor_2d_to_prediction_batch(t : torch.Tensor) :
    # Please note: on Cpp code PredictionBatch is constructed from 3d tensors
    # whose second dimension is 1, here it's constructed from 2d tensors
    pred = PredictionBatch()

    pred.b = t.shape[0]
    pred.cls = t.shape[1]
    pred.prob_matrix = t.softmax(dim=-1)

    pred.prob, pred.classes = torch.max(pred.prob_matrix, 1)

    return pred

def plot_prediction_batch(pred : PredictionBatch) :
    b, cls = pred.b, pred.cls
    print(f'Prediction[{b}], {cls} classes:')

    for i in range(b) :
        print('   ', end='')
        print(f'B[{i}]:', end=' ')
        print(f'class {pred.classes[i].item()}', end=', ')
        print(f'prob {pred.prob[i].item():7.3f}')
    print('')

    print('   ', end='')
    print(f'Probability Matrix[{b}x{cls}]:')
    for i in range(b) :
        print('   ', end='')
        for j in range(cls) :
            #print(f'{pred.prob_matrix[i, j].item():7.3f}', end=' ')
            print(f'{pred.prob_matrix[i, j].item():13.9f}', end=' ')
        print('')
    print('')



#
# Load and store from bytearray
#

LABEL_UNSIGNED = 'I'
LABEL_FLOAT = 'f'
LABEL_BOOL = '?'

SIZEOF_UNSIGNED = 4
SIZEOF_FLOAT = 4
SIZEOF_BOOL = 1

def append_tensor_to_bytearray(t : torch.Tensor, ba : bytearray) :
    ba.extend( bytearray(struct.pack(LABEL_UNSIGNED, t.dim())) )

    for sh in t.shape :
        ba.extend( bytearray(struct.pack(LABEL_UNSIGNED, sh)) )

    data = torch.reshape(t, (-1,)).tolist()
    for el in data :
        ba.extend( bytearray(struct.pack(LABEL_FLOAT, el)) )

    return ba

def extract_tensor_from_bytearray(ba : bytearray) :
    dim = struct.unpack(LABEL_UNSIGNED, ba[0:SIZEOF_UNSIGNED])
    dim = dim[0]
    ba = ba[SIZEOF_UNSIGNED : ]

    shape = []
    tot_elements = 1
    for i in range (dim) :
        sh = struct.unpack(LABEL_UNSIGNED, ba[i*SIZEOF_UNSIGNED : (i+1)*SIZEOF_UNSIGNED])
        sh = sh[0]
        tot_elements *= sh
        shape.append(sh)
    ba = ba[SIZEOF_UNSIGNED*dim : ]

    data = []
    for i in range (tot_elements) :
        el = struct.unpack(LABEL_FLOAT, ba[i*SIZEOF_FLOAT : (i+1)*SIZEOF_FLOAT])
        el = el[0]
        data.append(el)
    ba = ba[SIZEOF_FLOAT*tot_elements : ]

    t = torch.Tensor(data)
    t = torch.reshape(t, shape)

    return t, ba

def append_prediction_to_bytearray(pred : PredictionBatch, ba : bytearray) :
    ba.extend( bytearray(struct.pack(LABEL_UNSIGNED, pred.b)) )
    ba.extend( bytearray(struct.pack(LABEL_UNSIGNED, pred.cls)) )

    classes = pred.classes.tolist()
    for el in classes :
        ba.extend( bytearray(struct.pack(LABEL_UNSIGNED, el)) )

    prob = pred.prob.tolist()
    for el in prob :
        ba.extend( bytearray(struct.pack(LABEL_FLOAT, el)) )

    prob_matrix = torch.reshape(pred.prob_matrix, (-1,)).tolist()
    for el in prob_matrix :
        ba.extend( bytearray(struct.pack(LABEL_FLOAT, el)) )

    return ba

def extract_prediction_from_bytearray(ba : bytearray) :
    b = struct.unpack(LABEL_UNSIGNED, ba[0:SIZEOF_UNSIGNED])
    b = b[0]
    cls = struct.unpack(LABEL_UNSIGNED, ba[SIZEOF_UNSIGNED:2*SIZEOF_UNSIGNED])
    cls = cls[0]
    ba = ba[2*SIZEOF_UNSIGNED : ]

    classes = []
    for i in range (b) :
        el = struct.unpack(LABEL_UNSIGNED, ba[i*SIZEOF_UNSIGNED : (i+1)*SIZEOF_UNSIGNED])
        el = el[0]
        classes.append(el)
    ba = ba[SIZEOF_UNSIGNED*b : ]

    prob = []
    for i in range (b) :
        el = struct.unpack(LABEL_FLOAT, ba[i*SIZEOF_FLOAT : (i+1)*SIZEOF_FLOAT])
        el = el[0]
        prob.append(el)
    ba = ba[SIZEOF_FLOAT*b : ]

    prob_matrix = []
    for i in range (b*cls) :
        el = struct.unpack(LABEL_FLOAT, ba[i*SIZEOF_FLOAT : (i+1)*SIZEOF_FLOAT])
        el = el[0]
        prob_matrix.append(el)
    ba = ba[SIZEOF_FLOAT*b*cls : ]

    pred = PredictionBatch()
    pred.b = b
    pred.cls = cls
    pred.classes = torch.Tensor(classes).int()
    pred.prob = torch.Tensor(prob)
    pred.prob_matrix = torch.reshape(torch.Tensor(prob_matrix), (b, cls))

    return pred, ba

def append_linear_to_bytearray(lin : nn.Linear, ba : bytearray) :
    ba.extend( bytearray(struct.pack(LABEL_UNSIGNED, lin.in_features)) )
    ba.extend( bytearray(struct.pack(LABEL_UNSIGNED, lin.out_features)) )
    use_bias = False if lin.bias == None else True
    ba.extend( bytearray(struct.pack(LABEL_BOOL, use_bias)) )

    ba = append_tensor_to_bytearray(lin.weight.data, ba)
    if (use_bias) :
        ba = append_tensor_to_bytearray(lin.bias.data, ba)

    return ba

def extract_linear_from_bytearray(ba : bytearray) :
    in_features = struct.unpack(LABEL_UNSIGNED, ba[0:SIZEOF_UNSIGNED])
    in_features = in_features[0]
    out_features = struct.unpack(LABEL_UNSIGNED, ba[SIZEOF_UNSIGNED : 2*SIZEOF_UNSIGNED])
    out_features = out_features[0]
    use_bias = struct.unpack(LABEL_BOOL, ba[2*SIZEOF_UNSIGNED : 2*SIZEOF_UNSIGNED+SIZEOF_BOOL])
    use_bias = use_bias[0]
    ba = ba[2*SIZEOF_UNSIGNED+SIZEOF_BOOL : ]

    lin = nn.Linear(in_features=in_features, out_features=out_features, bias=use_bias)

    A, ba = extract_tensor_from_bytearray(ba)
    lin.weight.data = A

    if (use_bias) :
        b, ba = extract_tensor_from_bytearray(ba)
        lin.bias.data = b
    else :
        lin.bias = None

    return lin, ba

def append_layernorm_to_bytearray(ln : nn.LayerNorm, ba : bytearray) :
    ba.extend( bytearray(struct.pack(LABEL_UNSIGNED, ln.normalized_shape[0])) )
    ba.extend( bytearray(struct.pack(LABEL_FLOAT, ln.eps)) )
    use_bias = False if ln.bias == None else True
    ba.extend( bytearray(struct.pack(LABEL_BOOL, use_bias)) )

    ba = append_tensor_to_bytearray(ln.weight.data, ba)
    if (use_bias) :
        ba = append_tensor_to_bytearray(ln.bias.data, ba)

    return ba

def extract_layernorm_from_bytearray(ba : bytearray) :
    normalized_shape = struct.unpack(LABEL_UNSIGNED, ba[0:SIZEOF_UNSIGNED])
    normalized_shape = normalized_shape[0]
    eps = struct.unpack(LABEL_FLOAT, ba[SIZEOF_UNSIGNED : SIZEOF_UNSIGNED+SIZEOF_FLOAT])
    eps = eps[0]
    use_bias = struct.unpack(LABEL_BOOL, ba[SIZEOF_UNSIGNED+SIZEOF_FLOAT : SIZEOF_UNSIGNED+SIZEOF_FLOAT+SIZEOF_BOOL])
    use_bias = use_bias[0]
    ba = ba[SIZEOF_UNSIGNED+SIZEOF_FLOAT+SIZEOF_BOOL : ]

    ln = nn.LayerNorm(normalized_shape=normalized_shape, eps=eps, bias=use_bias)

    g, ba = extract_tensor_from_bytearray(ba)
    ln.weight.data = g

    if (use_bias) :
        b, ba = extract_tensor_from_bytearray(ba)
        ln.bias.data = b
    else :
        ln.bias = None

    return ln, ba

def append_layerscale_to_bytearray(ls : LayerScale, ba : bytearray) :
    ba.extend( bytearray(struct.pack(LABEL_UNSIGNED, len(ls.gamma))) )
    ba.extend( bytearray(struct.pack(LABEL_FLOAT, ls.gamma[0])) )

    return ba

def extract_layerscale_from_bytearray(ba : bytearray) :
    dim = struct.unpack(LABEL_UNSIGNED, ba[0:SIZEOF_UNSIGNED])
    dim = dim[0]
    val = struct.unpack(LABEL_FLOAT, ba[SIZEOF_UNSIGNED : SIZEOF_UNSIGNED+SIZEOF_FLOAT])
    val = val[0]
    ba = ba[SIZEOF_UNSIGNED+SIZEOF_FLOAT : ]

    return LayerScale(dim=dim, init_values=val), ba

def append_mlp_to_bytearray(mlp : Mlp, ba : bytearray) :
    ba.extend( bytearray(struct.pack(LABEL_UNSIGNED, mlp.fc1.weight.data.shape[1])) )
    ba.extend( bytearray(struct.pack(LABEL_UNSIGNED, mlp.fc1.weight.data.shape[0])) )
    ba.extend( bytearray(struct.pack(LABEL_UNSIGNED, mlp.fc1.weight.data.shape[1])) )
    use_norm = False if type(mlp.norm) == nn.Identity else True
    ba.extend( bytearray(struct.pack(LABEL_BOOL, use_norm)) )

    ba = append_linear_to_bytearray(mlp.fc1, ba)

    if ( type(mlp.act) == nn.ReLU ) :
        act_type = 1
    elif ( type(mlp.act) == nn.GELU ) :
        act_type = 2
    else :
        act_type = 0
    ba.extend( bytearray(struct.pack(LABEL_UNSIGNED, act_type)) )

    if (use_norm) :
        ba = append_layernorm_to_bytearray(mlp.norm, ba)

    ba = append_linear_to_bytearray(mlp.fc2, ba)

    return ba

def extract_mlp_from_bytearray(ba : bytearray) :
    in_features = struct.unpack(LABEL_UNSIGNED, ba[0:SIZEOF_UNSIGNED])
    in_features = in_features[0]
    hidden_features = struct.unpack(LABEL_UNSIGNED, ba[SIZEOF_UNSIGNED:2*SIZEOF_UNSIGNED])
    hidden_features = hidden_features[0]
    out_features = struct.unpack(LABEL_UNSIGNED, ba[2*SIZEOF_UNSIGNED:3*SIZEOF_UNSIGNED])
    out_features = out_features[0]
    use_norm = struct.unpack(LABEL_BOOL, ba[3*SIZEOF_UNSIGNED : 3*SIZEOF_UNSIGNED+SIZEOF_BOOL])
    use_norm = use_norm[0]
    ba = ba[3*SIZEOF_UNSIGNED+SIZEOF_BOOL : ]

    fc1, ba = extract_linear_from_bytearray(ba)

    act_type = struct.unpack(LABEL_UNSIGNED, ba[0:SIZEOF_UNSIGNED])
    act_type = act_type[0]
    ba = ba[SIZEOF_UNSIGNED : ]
    if ( act_type == 1 ) :
        act = nn.ReLU
    elif ( act_type == 2 ) :
        act = nn.GELU
    else :
        act_type = 0
    assert act_type != 0

    if (use_norm) :
        norm, ba = extract_layernorm_from_bytearray(ba)
    else :
        norm = nn.Identity()

    fc2, ba = extract_linear_from_bytearray(ba)

    mlp = Mlp(in_features=in_features, hidden_features=hidden_features, out_features=out_features,
        act_layer=act, norm_layer=None, bias=True, drop=0.0, use_conv=False)
    mlp.fc1 = fc1
    mlp.norm = norm
    mlp.fc2 = fc2

    return mlp, ba

def append_conv2d_to_bytearray(c2d : nn.Conv2d, ba : bytearray) :
    ba.extend( bytearray(struct.pack(LABEL_UNSIGNED, c2d.in_channels)) )
    ba.extend( bytearray(struct.pack(LABEL_UNSIGNED, c2d.out_channels)) )
    ba.extend( bytearray(struct.pack(LABEL_UNSIGNED, c2d.kernel_size[0])) )
    ba.extend( bytearray(struct.pack(LABEL_UNSIGNED, c2d.kernel_size[1])) )
    ba.extend( bytearray(struct.pack(LABEL_UNSIGNED, c2d.stride[0])) )
    ba.extend( bytearray(struct.pack(LABEL_UNSIGNED, c2d.stride[1])) )
    use_bias = False if c2d.bias == None else True
    ba.extend( bytearray(struct.pack(LABEL_BOOL, use_bias)) )

    ba = append_tensor_to_bytearray(c2d.weight.data, ba)
    if (use_bias) :
        ba = append_tensor_to_bytearray(c2d.bias.data, ba)

    return ba

def extract_conv2d_from_bytearray(ba : bytearray) :
    in_channels = struct.unpack(LABEL_UNSIGNED, ba[0:SIZEOF_UNSIGNED])
    in_channels = in_channels[0]
    out_channels = struct.unpack(LABEL_UNSIGNED, ba[SIZEOF_UNSIGNED:2*SIZEOF_UNSIGNED])
    out_channels = out_channels[0]
    kernel_size_h = struct.unpack(LABEL_UNSIGNED, ba[2*SIZEOF_UNSIGNED:3*SIZEOF_UNSIGNED])
    kernel_size_h = kernel_size_h[0]
    kernel_size_w = struct.unpack(LABEL_UNSIGNED, ba[3*SIZEOF_UNSIGNED:4*SIZEOF_UNSIGNED])
    kernel_size_w = kernel_size_w[0]
    stride_h = struct.unpack(LABEL_UNSIGNED, ba[4*SIZEOF_UNSIGNED:5*SIZEOF_UNSIGNED])
    stride_h = stride_h[0]
    stride_w = struct.unpack(LABEL_UNSIGNED, ba[5*SIZEOF_UNSIGNED:6*SIZEOF_UNSIGNED])
    stride_w = stride_w[0]
    use_bias = struct.unpack(LABEL_BOOL, ba[6*SIZEOF_UNSIGNED : 6*SIZEOF_UNSIGNED+SIZEOF_BOOL])
    use_bias = use_bias[0]
    ba = ba[6*SIZEOF_UNSIGNED+SIZEOF_BOOL : ]

    c2d = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
        kernel_size=(kernel_size_h, kernel_size_w), stride=(stride_h, stride_w), bias=True)
    
    kernel, ba = extract_tensor_from_bytearray(ba)
    c2d.weight.data = kernel

    if (use_bias) :
        bias, ba = extract_tensor_from_bytearray(ba)
        c2d.bias.data = bias
    else :
        c2d.bias = None

    return c2d, ba

def split_qkv(qkv : nn.Linear) :
    use_bias = False if qkv.bias == None else True

    qkv_A = qkv.weight.data
    qkv_A = torch.reshape(qkv_A, (3, -1, qkv_A.shape[1]))
    q_A = qkv_A[0]
    k_A = qkv_A[1]
    v_A = qkv_A[2]

    q_gen = nn.Linear(in_features=q_A.shape[1], out_features=q_A.shape[0], bias=use_bias)
    q_gen.weight.data = q_A
    k_gen = nn.Linear(in_features=k_A.shape[1], out_features=k_A.shape[0], bias=use_bias)
    k_gen.weight.data = k_A
    v_gen = nn.Linear(in_features=v_A.shape[1], out_features=v_A.shape[0], bias=use_bias)
    v_gen.weight.data = v_A

    if (use_bias) :
        qkv_b = qkv.bias.data
        qkv_b = torch.reshape(qkv_b, (3, -1))
        q_b = qkv_b[0]
        k_b = qkv_b[1]
        v_b = qkv_b[2]

        q_gen.bias.data = q_b
        k_gen.bias.data = k_b
        v_gen.bias.data = v_b

    return q_gen, k_gen, v_gen

def append_attention_to_bytearray(attn : Attention, ba : bytearray) :
    ba.extend( bytearray(struct.pack(LABEL_UNSIGNED, attn.num_heads*attn.head_dim)) )
    ba.extend( bytearray(struct.pack(LABEL_UNSIGNED, attn.num_heads)) )
    ba.extend( bytearray(struct.pack(LABEL_UNSIGNED, attn.head_dim)) )
    ba.extend( bytearray(struct.pack(LABEL_FLOAT, attn.scale)) )
    use_qk_norm = False if type(attn.q_norm) == nn.Identity else True
    ba.extend( bytearray(struct.pack(LABEL_BOOL, use_qk_norm)) )

    q_gen, k_gen, v_gen = split_qkv(attn.qkv)
    ba = append_linear_to_bytearray(q_gen, ba)
    ba = append_linear_to_bytearray(k_gen, ba)
    ba = append_linear_to_bytearray(v_gen, ba)
    if (use_qk_norm) :
        ba = append_layernorm_to_bytearray(attn.q_norm, ba)
        ba = append_layernorm_to_bytearray(attn.k_norm, ba)
    ba = append_linear_to_bytearray(attn.proj, ba)

    return ba

def fuse_qkv(q_gen : nn.Linear, k_gen: nn.Linear, v_gen : nn.Linear) :
    use_bias = False if q_gen.bias == None else True

    q_A = q_gen.weight.data
    k_A = k_gen.weight.data
    v_A = v_gen.weight.data
    qkv_A = torch.cat((q_A, k_A, v_A), 0)

    qkv_gen = nn.Linear(in_features=qkv_A.shape[1], out_features=qkv_A.shape[0], bias=use_bias)
    qkv_gen.weight.data = qkv_A

    if (use_bias) :
        q_b = q_gen.bias.data
        k_b = k_gen.bias.data
        v_b = v_gen.bias.data
        qkv_b = torch.cat((q_b, k_b, v_b), 0)
        qkv_gen.bias.data = qkv_b

    return qkv_gen

def extract_attention_from_bytearray(ba : bytearray) :
    dim = struct.unpack(LABEL_UNSIGNED, ba[0:SIZEOF_UNSIGNED])
    dim = dim[0]
    num_heads = struct.unpack(LABEL_UNSIGNED, ba[SIZEOF_UNSIGNED:2*SIZEOF_UNSIGNED])
    num_heads = num_heads[0]
    head_dim = struct.unpack(LABEL_UNSIGNED, ba[2*SIZEOF_UNSIGNED:3*SIZEOF_UNSIGNED])
    head_dim = head_dim[0]
    scale = struct.unpack(LABEL_FLOAT, ba[3*SIZEOF_UNSIGNED:3*SIZEOF_UNSIGNED+SIZEOF_FLOAT])
    scale = scale[0]
    use_qk_norm = struct.unpack(LABEL_BOOL, ba[3*SIZEOF_UNSIGNED+SIZEOF_FLOAT : 3*SIZEOF_UNSIGNED+SIZEOF_FLOAT+SIZEOF_BOOL])
    use_qk_norm = use_qk_norm[0]
    ba = ba[3*SIZEOF_UNSIGNED+SIZEOF_FLOAT+SIZEOF_BOOL : ]

    attn = Attention(dim=dim, num_heads=num_heads, qkv_bias=True, qk_norm=use_qk_norm, attn_drop=0.0,
            proj_drop=0.0, norm_layer=nn.LayerNorm)

    q_gen, ba = extract_linear_from_bytearray(ba)
    k_gen, ba = extract_linear_from_bytearray(ba)
    v_gen, ba = extract_linear_from_bytearray(ba)
    qkv_gen = fuse_qkv(q_gen, k_gen, v_gen)
    attn.qkv = qkv_gen

    if (use_qk_norm) :
        q_norm, ba = extract_layernorm_from_bytearray(ba)
        k_norm, ba = extract_layernorm_from_bytearray(ba)
        attn.q_norm = q_norm
        attn.k_norm = k_norm
    else :
        attn.q_norm = nn.Identity()
        attn.k_norm = nn.Identity()

    proj, ba = extract_linear_from_bytearray(ba)
    attn.proj = proj

    return attn, ba

def append_block_to_bytearray(block : Block, ba : bytearray) :
    ba.extend( bytearray(struct.pack(LABEL_UNSIGNED, block.attn.num_heads * block.attn.head_dim)) )
    ba.extend( bytearray(struct.pack(LABEL_UNSIGNED, block.attn.num_heads)) )
    mlp_ratio = block.mlp.fc1.weight.data.shape[0] / block.mlp.fc1.weight.data.shape[1]
    ba.extend( bytearray(struct.pack(LABEL_FLOAT, mlp_ratio)) )

    ba = append_layernorm_to_bytearray(block.norm1, ba)
    ba = append_attention_to_bytearray(block.attn, ba)

    if ( type(block.ls1) == nn.Identity ) :
        ba = append_layerscale_to_bytearray(LayerScale(block.attn.num_heads * block.attn.head_dim, 1.0), ba)
    else :
        ba = append_layerscale_to_bytearray(block.ls1, ba)

    ba = append_layernorm_to_bytearray(block.norm2, ba)
    ba = append_mlp_to_bytearray(block.mlp, ba)

    if ( type(block.ls2) == nn.Identity ) :
        ba = append_layerscale_to_bytearray(LayerScale(block.attn.num_heads * block.attn.head_dim, 1.0), ba)
    else :
        ba = append_layerscale_to_bytearray(block.ls2, ba)

    return ba

def extract_block_from_bytearray(ba : bytearray) :
    dim = struct.unpack(LABEL_UNSIGNED, ba[0:SIZEOF_UNSIGNED])
    dim = dim[0]
    num_heads = struct.unpack(LABEL_UNSIGNED, ba[SIZEOF_UNSIGNED:2*SIZEOF_UNSIGNED])
    num_heads = num_heads[0]
    mlp_ratio = struct.unpack(LABEL_FLOAT, ba[2*SIZEOF_UNSIGNED:2*SIZEOF_UNSIGNED+SIZEOF_FLOAT])
    mlp_ratio = mlp_ratio[0]
    ba = ba[2*SIZEOF_UNSIGNED+SIZEOF_FLOAT : ]

    block = Block(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio)

    norm1, ba = extract_layernorm_from_bytearray(ba)
    attn, ba = extract_attention_from_bytearray(ba)
    ls1, ba = extract_layerscale_from_bytearray(ba)
    norm2, ba = extract_layernorm_from_bytearray(ba)
    mlp, ba = extract_mlp_from_bytearray(ba)
    ls2, ba = extract_layerscale_from_bytearray(ba)

    block.norm1 = norm1
    block.attn = attn
    block.ls1 = ls1
    block.norm2 = norm2
    block.mlp = mlp
    block.ls2 = ls2

    return block, ba

def append_patch_embed_to_bytearray(pe : PatchEmbed, ba : bytearray) :
    ba.extend( bytearray(struct.pack(LABEL_UNSIGNED, pe.img_size[0])) )
    ba.extend( bytearray(struct.pack(LABEL_UNSIGNED, pe.img_size[1])) )
    ba.extend( bytearray(struct.pack(LABEL_UNSIGNED, pe.patch_size[0])) )
    ba.extend( bytearray(struct.pack(LABEL_UNSIGNED, pe.patch_size[1])) )

    ba.extend( bytearray(struct.pack(LABEL_UNSIGNED, pe.grid_size[0])) )
    ba.extend( bytearray(struct.pack(LABEL_UNSIGNED, pe.grid_size[1])) )
    ba.extend( bytearray(struct.pack(LABEL_UNSIGNED, pe.num_patches)) )

    ba.extend( bytearray(struct.pack(LABEL_UNSIGNED, pe.proj.weight.data.shape[1])) )
    ba.extend( bytearray(struct.pack(LABEL_UNSIGNED, pe.proj.weight.data.shape[0])) )

    ba.extend( bytearray(struct.pack(LABEL_BOOL, pe.strict_img_size)) )
    ba.extend( bytearray(struct.pack(LABEL_BOOL, pe.dynamic_img_pad)) )
    use_norm = False if type(pe.norm) == nn.Identity else True
    ba.extend( bytearray(struct.pack(LABEL_BOOL, use_norm)) )

    ba = append_conv2d_to_bytearray(pe.proj, ba)
    if (use_norm) :
        ba = append_layernorm_to_bytearray(pe.norm, ba)

    return ba

def extract_patch_embed_from_bytearray(ba : bytearray) :
    img_size_h = struct.unpack(LABEL_UNSIGNED, ba[0:SIZEOF_UNSIGNED])
    img_size_h = img_size_h[0]
    img_size_w = struct.unpack(LABEL_UNSIGNED, ba[SIZEOF_UNSIGNED:2*SIZEOF_UNSIGNED])
    img_size_w = img_size_w[0]
    patch_size_h = struct.unpack(LABEL_UNSIGNED, ba[2*SIZEOF_UNSIGNED:3*SIZEOF_UNSIGNED])
    patch_size_h = patch_size_h[0]
    patch_size_w = struct.unpack(LABEL_UNSIGNED, ba[3*SIZEOF_UNSIGNED:4*SIZEOF_UNSIGNED])
    patch_size_w = patch_size_w[0]

    grid_size_h = struct.unpack(LABEL_UNSIGNED, ba[4*SIZEOF_UNSIGNED:5*SIZEOF_UNSIGNED])
    grid_size_h = grid_size_h[0]
    grid_size_w = struct.unpack(LABEL_UNSIGNED, ba[5*SIZEOF_UNSIGNED:6*SIZEOF_UNSIGNED])
    grid_size_w = grid_size_w[0]
    num_patches = struct.unpack(LABEL_UNSIGNED, ba[6*SIZEOF_UNSIGNED:7*SIZEOF_UNSIGNED])
    num_patches = num_patches[0]

    in_chans = struct.unpack(LABEL_UNSIGNED, ba[7*SIZEOF_UNSIGNED:8*SIZEOF_UNSIGNED])
    in_chans = in_chans[0]
    embed_dim = struct.unpack(LABEL_UNSIGNED, ba[8*SIZEOF_UNSIGNED:9*SIZEOF_UNSIGNED])
    embed_dim = embed_dim[0]
    ba = ba[9*SIZEOF_UNSIGNED : ]

    strict_img_size = struct.unpack(LABEL_BOOL, ba[0:SIZEOF_BOOL])
    strict_img_size = strict_img_size[0]
    dynamic_img_pad = struct.unpack(LABEL_BOOL, ba[SIZEOF_BOOL:2*SIZEOF_BOOL])
    dynamic_img_pad = dynamic_img_pad[0]
    use_norm = struct.unpack(LABEL_BOOL, ba[2*SIZEOF_BOOL:3*SIZEOF_BOOL])
    use_norm = use_norm[0]
    ba = ba[3*SIZEOF_BOOL : ]

    pe = PatchEmbed(img_size=img_size_h, patch_size=patch_size_h, in_chans=in_chans,
        embed_dim=embed_dim, norm_layer=None, flatten=True, bias=True,
        strict_img_size=strict_img_size, dynamic_img_pad=dynamic_img_pad)
    pe.img_size = (img_size_h, img_size_w)
    pe.patch_size = (patch_size_h, patch_size_w)
    pe.grid_size = (grid_size_h, grid_size_w)
    pe.num_patches = num_patches

    proj, ba = extract_conv2d_from_bytearray(ba)
    pe.proj = proj

    if (use_norm) :
        norm, ba = extract_layernorm_from_bytearray(ba)
        pe.norm = norm
    else :
        pe.norm = nn.Identity()

    return pe, ba

def append_vit_to_bytearray(vit : VisionTransformer, ba : bytearray) :
    ba.extend( bytearray(struct.pack(LABEL_UNSIGNED, vit.num_classes)) )

    if ( vit.global_pool == 'token' ) :
        pool_type = 1
    elif ( vit.global_pool == 'avg' ) :
        pool_type = 2
    elif ( vit.global_pool == 'avgmax' ) :
        pool_type = 3
    elif ( vit.global_pool == 'max' ) :
        pool_type = 4
    elif ( vit.global_pool == 'map' ) :
        pool_type = 5
    else :
        pool_type = 0
    ba.extend( bytearray(struct.pack(LABEL_UNSIGNED, pool_type)) )

    ba.extend( bytearray(struct.pack(LABEL_UNSIGNED, vit.embed_dim)) )
    ba.extend( bytearray(struct.pack(LABEL_UNSIGNED, len(vit.blocks))) )

    ba.extend( bytearray(struct.pack(LABEL_BOOL, vit.has_class_token)) )
    ba.extend( bytearray(struct.pack(LABEL_UNSIGNED, vit.num_reg_tokens)) )
    ba.extend( bytearray(struct.pack(LABEL_UNSIGNED, vit.num_prefix_tokens)) )
    ba.extend( bytearray(struct.pack(LABEL_BOOL, vit.no_embed_class)) )

    use_pos_embed = False if vit.pos_embed == None else True
    ba.extend( bytearray(struct.pack(LABEL_BOOL, use_pos_embed)) )
    use_pre_norm = False if type(vit.norm_pre) == nn.Identity else True
    ba.extend( bytearray(struct.pack(LABEL_BOOL, use_pre_norm)) )
    use_fc_norm = False if type(vit.fc_norm) == nn.Identity else True
    ba.extend( bytearray(struct.pack(LABEL_BOOL, use_fc_norm)) )
    ba.extend( bytearray(struct.pack(LABEL_BOOL, vit.dynamic_img_size)) )

    if (vit.has_class_token) :
        ba = append_tensor_to_bytearray(vit.cls_token.data[0, 0], ba)
    if (vit.num_reg_tokens > 0) :
        ba = append_tensor_to_bytearray(vit.reg_token.data[0], ba)
    if (use_pos_embed) :
        ba = append_tensor_to_bytearray(vit.pos_embed.data[0], ba)

    ba = append_patch_embed_to_bytearray(vit.patch_embed, ba)
    if (use_pre_norm) :
        ba = append_layernorm_to_bytearray(vit.norm_pre, ba)

    for block in vit.blocks :
        ba = append_block_to_bytearray(block, ba)

    if (use_fc_norm) :
        ba = append_layernorm_to_bytearray(vit.fc_norm, ba)
    else :
        ba = append_layernorm_to_bytearray(vit.norm, ba)
    ba = append_linear_to_bytearray(vit.head, ba)

    return ba

def extract_vit_from_bytearray(ba : bytearray) :
    num_classes = struct.unpack(LABEL_UNSIGNED, ba[0:SIZEOF_UNSIGNED])
    num_classes = num_classes[0]

    pool_type = struct.unpack(LABEL_UNSIGNED, ba[SIZEOF_UNSIGNED:2*SIZEOF_UNSIGNED])
    pool_type = pool_type[0]
    if ( pool_type == 1 ) :
        pool_type = 'token'
    elif ( pool_type == 2 ) :
        pool_type = 'avg'
    elif ( pool_type == 3 ) :
        pool_type = 'avgmax'
    elif ( pool_type == 4 ) :
        pool_type = 'max'
    elif ( pool_type == 5 ) :
        pool_type = 'map'
    else :
        pool_type = ''

    embed_dim = struct.unpack(LABEL_UNSIGNED, ba[2*SIZEOF_UNSIGNED:3*SIZEOF_UNSIGNED])
    embed_dim = embed_dim[0]
    depth = struct.unpack(LABEL_UNSIGNED, ba[3*SIZEOF_UNSIGNED:4*SIZEOF_UNSIGNED])
    depth = depth[0]
    ba = ba[4*SIZEOF_UNSIGNED : ]

    has_class_token = struct.unpack(LABEL_BOOL, ba[0:SIZEOF_BOOL])
    has_class_token = has_class_token[0]
    num_reg_tokens = struct.unpack(LABEL_UNSIGNED, ba[SIZEOF_BOOL:SIZEOF_BOOL+SIZEOF_UNSIGNED])
    num_reg_tokens = num_reg_tokens[0]
    num_prefix_tokens = struct.unpack(LABEL_UNSIGNED, ba[SIZEOF_BOOL+SIZEOF_UNSIGNED:SIZEOF_BOOL+2*SIZEOF_UNSIGNED])
    num_prefix_tokens = num_prefix_tokens[0]
    no_embed_class = struct.unpack(LABEL_BOOL, ba[SIZEOF_BOOL+2*SIZEOF_UNSIGNED:2*SIZEOF_BOOL+2*SIZEOF_UNSIGNED])
    no_embed_class = no_embed_class[0]
    ba = ba[2*SIZEOF_BOOL+2*SIZEOF_UNSIGNED : ]

    use_pos_embed = struct.unpack(LABEL_BOOL, ba[0:SIZEOF_BOOL])
    use_pos_embed = use_pos_embed[0]
    use_pre_norm = struct.unpack(LABEL_BOOL, ba[SIZEOF_BOOL:2*SIZEOF_BOOL])
    use_pre_norm = use_pre_norm[0]
    use_fc_norm = struct.unpack(LABEL_BOOL, ba[2*SIZEOF_BOOL:3*SIZEOF_BOOL])
    use_fc_norm = use_fc_norm[0]
    dynamic_img_size = struct.unpack(LABEL_BOOL, ba[3*SIZEOF_BOOL:4*SIZEOF_BOOL])
    dynamic_img_size = dynamic_img_size[0]
    ba = ba[4*SIZEOF_BOOL : ]

    if (has_class_token) :
        cls_token, ba = extract_tensor_from_bytearray(ba)
        cls_token = torch.reshape(cls_token, (1, 1, cls_token.shape[0]))
    else :
        cls_token = None
    if (num_reg_tokens > 0) :
        reg_token, ba = extract_tensor_from_bytearray(ba)
        reg_token = torch.reshape(reg_token, (1, reg_token.shape[0], reg_token.shape[1]))
    else :
        reg_token = None
    if (use_pos_embed) :
        pos_embed, ba = extract_tensor_from_bytearray(ba)
        pos_embed = torch.reshape(pos_embed, (1, pos_embed.shape[0], pos_embed.shape[1]))
    else :
        pos_embed = None

    patch_embed, ba = extract_patch_embed_from_bytearray(ba)
    if (use_pre_norm) :
        norm_pre, ba = extract_layernorm_from_bytearray(ba)
    else :
        norm_pre = nn.Identity()

    blocks = nn.Sequential()
    for i in range(depth) :
        block, ba = extract_block_from_bytearray(ba)
        blocks.append(block)

    if (use_fc_norm) :
        norm = nn.Identity()
        fc_norm, ba = extract_layernorm_from_bytearray(ba)
    else :
        norm, ba = extract_layernorm_from_bytearray(ba)
        fc_norm = nn.Identity()
    head, ba = extract_linear_from_bytearray(ba)

    vit = VisionTransformer(
        img_size = patch_embed.img_size,
        patch_size = patch_embed.patch_size,
        in_chans = patch_embed.proj.weight.data.shape[1],
        num_classes = num_classes,
        global_pool = pool_type,
        embed_dim = embed_dim,
        depth = depth,
        num_heads = blocks[0].attn.num_heads,
        mlp_ratio = blocks[0].mlp.fc1.weight.data.shape[0] / blocks[0].mlp.fc1.weight.data.shape[1],
        qkv_bias = True,
        qk_norm = False,
        init_values = None,
        class_token = has_class_token,
        pos_embed = 'learn' if use_pos_embed else 'none',
        no_embed_class = no_embed_class,
        reg_tokens = num_reg_tokens,
        pre_norm = use_pre_norm,
        fc_norm = None,
        dynamic_img_size = dynamic_img_size,
        dynamic_img_pad = patch_embed.dynamic_img_pad,
        drop_rate = 0.0,
        pos_drop_rate = 0.0,
        patch_drop_rate = 0.0,
        proj_drop_rate = 0.0,
        attn_drop_rate = 0.0,
        drop_path_rate = 0.0,
        weight_init = '',
        fix_init = False,
        embed_layer = PatchEmbed,
        norm_layer = None,
        act_layer = None,
        block_fn = Block,
        mlp_layer = Mlp
    )

    if (has_class_token) :
        vit.cls_token = nn.Parameter(cls_token)
    else :
        vit.cls_token = None
    if (num_reg_tokens > 0) :
        vit.reg_token = nn.Parameter(reg_token)
    else :
        vit.reg_token = None
    if (use_pos_embed) :
        vit.pos_embed = nn.Parameter(pos_embed)
    else :
        vit.pos_embed = None

    vit.patch_embed = patch_embed
    vit.norm_pre = norm_pre
    vit.blocks = blocks
    vit.norm = norm
    vit.fc_norm = fc_norm
    vit.head = head

    return vit, ba



#
# Load and store from file
#

def store_cvit(vit : VisionTransformer, path : str) :
    ba = bytearray(b'CVIT')

    ba = append_vit_to_bytearray(vit, ba)

    file = open(path, 'wb')
    file.write(ba)
    file.close()

def load_cvit(path : str) :
    file = open(path, 'rb')
    ba = bytearray( file.read() )
    file.close()

    cvit_label = ba[0:4]
    ba = ba[4 : ]
    assert cvit_label == b'CVIT'

    vit, ba = extract_vit_from_bytearray(ba)
    return vit

def store_cpic(pic : torch.Tensor, path : str) :
    ba = bytearray(b'CPIC')

    ba = append_tensor_to_bytearray(pic, ba)

    file = open(path, 'wb')
    file.write(ba)
    file.close()

def load_cpic(path : str) :
    file = open(path, 'rb')
    ba = bytearray( file.read() )
    file.close()

    cvit_label = ba[0:4]
    ba = ba[4 : ]
    assert cvit_label == b'CPIC'

    pic, ba = extract_tensor_from_bytearray(ba)
    return pic

def store_cprd(pred : PredictionBatch, path : str) :
    ba = bytearray(b'CPRD')

    ba = append_prediction_to_bytearray(pred, ba)

    file = open(path, 'wb')
    file.write(ba)
    file.close()

def load_cprd(path : str) :
    file = open(path, 'rb')
    ba = bytearray( file.read() )
    file.close()

    cvit_label = ba[0:4]
    ba = ba[4 : ]
    assert cvit_label == b'CPRD'

    pred, ba = extract_prediction_from_bytearray(ba)
    return pred
