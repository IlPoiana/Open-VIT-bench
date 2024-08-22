#include "../include/modules.h"

#include <iostream>

using namespace std;

int main() {
    cout << "Test Modules" << endl;

    // Linear Test

    vit_float A_data[6*4] = {
        -81.543,  80.500,  75.219,  28.253,
        -81.939, -24.951,  70.828,  95.912,
        -82.804,   5.749,  -7.031, -28.139,
        -78.044,  16.657,  37.790,  21.834,
         89.140,  74.909,   9.010, -17.762,
         -4.832,  33.422,  86.924,  73.275
    };
    Matrix A(A_data, 6*4, 6, 4);
    cout << "### A" << endl;
    A.print();

    vit_float b_data[6] = {51.531, 67.995, 17.472, 95.498, -62.058, -12.408};
    RowVector b(b_data, 6);
    cout << "### b" << endl;
    b.print();

    Linear lin(4, 6, true);
    lin.move_A(A);
    lin.move_b(b);

    vit_float x_data[3*7*4] = {
        -18.674, -44.552, -65.284,   3.089,
         31.012,   4.975,  44.711,  -5.537,
        -34.826,  63.405,  -7.124,   3.944,
        -25.378,  51.703,  -5.950, -99.741,
        -10.568, -66.795, -42.139,  68.497,
        -96.361, -20.004,  -1.971,  93.979,
         82.150,   7.553, -74.959,  87.741,

         18.294,  71.665,  87.329,  81.982,
        -70.045, -63.034,  65.573, -47.865,
        -72.120, -39.214,  81.975, -56.392,
         77.278,  53.481, -96.486, -59.986,
         -9.931,  80.494, -47.813, -96.357,
        -92.930,  71.853,  89.731, -53.027,
         87.477, -29.208, -24.116, -46.039,

        -89.289, -22.255,  94.870,  24.519,
         13.026,   0.482,  72.868, -87.957,
         12.995, -33.355, -30.151,  98.456,
        -53.378,  93.617, -48.623, -12.992,
         60.356, -58.934,  93.257, -87.589,
         16.750,  36.116, -35.526,  71.693,
        -21.452, -66.819, -34.460,  26.530
    };
    Tensor x(x_data, 3*7*4, 3, 7, 4);
    cout << "### x" << endl;
    x.print();

    Tensor y;
    lin(x, y);
    cout << "### y = lin(x)" << endl;
    y.print();

    // LayerNorm Test

    vit_float g_data[6] = {-87.035, 39.796, 69.303, -97.629, 34.223, 63.169};
    RowVector g(g_data, 6);
    cout << "### g" << endl;
    g.print();

    vit_float b2_data[6] = {71.448, -20.092, -75.566, 6.899, 56.601, 16.178};
    RowVector b2(b2_data, 6);
    cout << "### b2" << endl;
    b2.print();

    LayerNorm ln(6, 0.00001, true);
    ln.move_g(g);
    ln.move_b(b2);

    ln(y);
    cout << "### normalized y" << endl;
    y.print();

    // MultiHead LayerNorm Test

    vit_float mhg_data[3] = {97.805, 19.679, 36.741};
    RowVector mh_g(mhg_data, 3);
    cout << "### multi-head gamma" << endl;
    mh_g.print();

    LayerNorm ln2(3, 0.00001, false);
    ln2.move_g(mh_g);

    ln2(y, 2, 3);
    cout << "### multi-head(2,3) normalized y" << endl;
    y.print();

    // LayerScale Test

    LayerScale ls(6, 3.125);
    cout << "### LayerScale val" << endl;
    cout << ls.get_val() << endl << endl;

    ls(y);
    cout << "### y rescaled by 3.125" << endl;
    y.print();

    // Activation Test

    Activation act_gelu(GELU);
    act_gelu(y);
    cout << "### GELU(y)" << endl;
    y.print();

    Activation act_relu(ReLU);
    act_relu(y);
    cout << "### ReLU(y)" << endl;
    y.print();

    // Global Pool Test

    Tensor z;
    global_pool_nlc(y, z, pool_token, 1, false);
    cout << "### z = pool_token(y)" << endl;
    z.print();

    global_pool_nlc(y, z, pool_avg, 1, false);
    cout << "### z = pool_avg(y)" << endl;
    z.print();

    global_pool_nlc(y, z, pool_max, 1, false);
    cout << "### z = pool_max(y)" << endl;
    z.print();
    
    global_pool_nlc(y, z, pool_avgmax, 1, false);
    cout << "### z = pool_avgmax(y)" << endl;
    z.print();

    return 0;
}
