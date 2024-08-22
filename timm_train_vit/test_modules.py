import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cvit_utils import plot_tensor
from timm.models.vision_transformer import LayerScale, global_pool_nlc

print("Test Modules")

# Linear Test

lin = nn.Linear(in_features=4, out_features=6, bias=True)
print("### initial random weight")
plot_tensor(lin.weight.data)
print("### initial random bias")
plot_tensor(lin.bias.data)

w = torch.tensor([[-81.543,  80.500,  75.219,  28.253],
                  [-81.939, -24.951,  70.828,  95.912],
                  [-82.804,   5.749,  -7.031, -28.139],
                  [-78.044,  16.657,  37.790,  21.834],
                  [ 89.140,  74.909,   9.010, -17.762],
                  [ -4.832,  33.422,  86.924,  73.275]])
lin.weight.data = w
print("### new weight")
plot_tensor(lin.weight.data)

b = torch.tensor([51.531, 67.995, 17.472, 95.498, -62.058, -12.408])
lin.bias.data = b
print("### new bias")
plot_tensor(lin.bias.data)

x = torch.tensor([[[-18.674, -44.552, -65.284,   3.089],
                   [ 31.012,   4.975,  44.711,  -5.537],
                   [-34.826,  63.405,  -7.124,   3.944],
                   [-25.378,  51.703,  -5.950, -99.741],
                   [-10.568, -66.795, -42.139,  68.497],
                   [-96.361, -20.004,  -1.971,  93.979],
                   [ 82.150,   7.553, -74.959,  87.741]],

                  [[ 18.294,  71.665,  87.329,  81.982],
                   [-70.045, -63.034,  65.573, -47.865],
                   [-72.120, -39.214,  81.975, -56.392],
                   [ 77.278,  53.481, -96.486, -59.986],
                   [ -9.931,  80.494, -47.813, -96.357],
                   [-92.930,  71.853,  89.731, -53.027],
                   [ 87.477, -29.208, -24.116, -46.039]],

                  [[-89.289, -22.255,  94.870,  24.519],
                   [ 13.026,   0.482,  72.868, -87.957],
                   [ 12.995, -33.355, -30.151,  98.456],
                   [-53.378,  93.617, -48.623, -12.992],
                   [ 60.356, -58.934,  93.257, -87.589],
                   [ 16.750,  36.116, -35.526,  71.693],
                   [-21.452, -66.819, -34.460,  26.530]]])
print("### x")
plot_tensor(x)

y = lin(x)
print("### y = lin(x)")
plot_tensor(y)

# LayerNorm Test

ln = nn.LayerNorm(normalized_shape=6, eps=0.00001, bias=True)
print("### initial random gamma")
plot_tensor(ln.weight.data)
print("### initial random bias")
plot_tensor(ln.bias.data)

g = torch.tensor([-87.035, 39.796, 69.303, -97.629, 34.223, 63.169])
ln.weight.data = g
print("### new gamma")
plot_tensor(ln.weight.data)

b2 = torch.tensor([71.448, -20.092, -75.566, 6.899, 56.601, 16.178])
ln.bias.data = b2
print("### new bias")
plot_tensor(ln.bias.data)

y = ln(y)
print("### normalized y")
plot_tensor(y)

# MultiHead LayerNorm Test

mh_g = torch.tensor([97.805, 19.679, 36.741])

ln2 = nn.LayerNorm(normalized_shape=3, eps=0.00001, bias=False)
ln2.weight.data = mh_g
print("### multi-head gamma")
plot_tensor(ln2.weight.data)

y = y.reshape(3, 7, 2, 3) # split 6 channles in 3x2
print("### reshaped y")
plot_tensor(y)
y = ln2(y)
print("### normalized reshaped y")
plot_tensor(y)
y = y.reshape(3, 7, 6) # split 6 channles in 3x2
print("### multi-head(2,3) normalized y")
plot_tensor(y)

# LayerScale Test

ls = LayerScale(dim=6, init_values=3.125, inplace=False)
print("### LayerScale val")
plot_tensor(ls.gamma)

y = ls(y)
print("### y rescaled by 3.125")
plot_tensor(y)

# Activation Test

act_gelu = nn.GELU(approximate='tanh')
y = act_gelu(y)
print("### GELU(y)")
plot_tensor(y)

act_relu = nn.ReLU()
y = act_relu(y)
print("### ReLU(y)")
plot_tensor(y)

# Global Pool Test

z = global_pool_nlc(y, pool_type='token', num_prefix_tokens=1, reduce_include_prefix=False)
print("### z = pool_token(y)")
plot_tensor(z)

z = global_pool_nlc(y, pool_type='avg', num_prefix_tokens=1, reduce_include_prefix=False)
print("### z = pool_avg(y)")
plot_tensor(z)

z = global_pool_nlc(y, pool_type='max', num_prefix_tokens=1, reduce_include_prefix=False)
print("### z = pool_max(y)")
plot_tensor(z)

z = global_pool_nlc(y, pool_type='avgmax', num_prefix_tokens=1, reduce_include_prefix=False)
print("### z = pool_avgmax(y)")
plot_tensor(z)
