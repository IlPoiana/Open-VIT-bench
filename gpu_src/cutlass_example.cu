#include "../cutlass/include/cutlass/cutlass.h"
#include "../cutlass/include/cutlass/gemm/device/gemm_universal.h"
#include "../cutlass/include/cutlass/epilogue/thread/linear_combination_generic.h"
#include "../cutlass/include/cutlass/epilogue/thread/activation.h"
#include "../cutlass/include/cutlass/layout/layout.h"
#include "../cutlass/tools/util/include/cutlass/util/host_tensor.h"
#include "../cutlass/tools/util/include/cutlass/util/reference/device/gemm.h"
#include <iostream>

#include "../cutlass/include/cutlass/epilogue/thread/linear_combination_gelu.h"


using ElementInputA = cutlass::half_t;
using ElementInputB = cutlass::half_t;
using ElementOutput = cutlass::half_t;
using ElementAccumulator = float;

// All row-major (no transposition)
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::RowMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

// ---------------------------------------------------------------------------
// 1. Define Epilogue: bias add then GELU (exact order)
// ---------------------------------------------------------------------------

// We'll use a custom "LinearCombinationGeneric" with GELU activation.
// The input to GELU is (acc + bias), computed in float.
using EpilogueOp = cutlass::epilogue::thread::LinearCombinationGeneric<
    cutlass::epilogue::thread::BiasAdd,          // first add bias
    cutlass::epilogue::thread::GELU,             // then apply GELU
    ElementOutput,                               // element output type
    8,                                           // elements per access
    ElementAccumulator,                          // accumulator type
    ElementAccumulator                           // compute type
>;

using GeluOp = cutlass::epilogue::thread::LinearCombinationGELU<
    ElementOutput,                   // Element output
    8,                               // Elements per vectorized access
    ElementAccumulator,          // Accumulator type
    ElementAccumulator           // Compute type
>;

// ---------------------------------------------------------------------------
// 2. Define a SIMT GEMM (no tensor cores)
// ---------------------------------------------------------------------------

using Gemm = cutlass::gemm::device::GemmUniversal<
    ElementInputA,
    LayoutA,
    ElementInputB,
    LayoutB,
    ElementOutput,
    LayoutD,
    ElementAccumulator,
    EpilogueOp,
    cutlass::arch::OpClassSimt,    // ✅ force SIMT (no tensor cores)
    cutlass::arch::Sm80            // ✅ good for CUDA 12.1.1+
>;

int main() {
    int M = 128, N = 256, K = 512;

    cutlass::HostTensor<ElementInputA, LayoutA> A({M, K});
    cutlass::HostTensor<ElementInputB, LayoutB> B({K, N});
    cutlass::HostTensor<ElementOutput, LayoutC> C({M, N});
    cutlass::HostTensor<ElementOutput, LayoutD> D({M, N});
    cutlass::HostTensor<ElementAccumulator, LayoutD> bias({1, N});

    A.fill_random(1);
    B.fill_random(1);
    C.fill_random(1);
    bias.fill_random(1);

    A.sync_device();
    B.sync_device();
    C.sync_device();
    bias.sync_device();

    typename Gemm::Arguments args{
        {M, N, K},
        {A.device_data(), K},
        {B.device_data(), N},
        {C.device_data(), N},
        {D.device_data(), N},
        {1.0f, 0.0f},               // alpha, beta
        bias.device_data()          // bias pointer passed to epilogue
    };

    Gemm gemm_op;
    cutlass::Status status = gemm_op(args);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "GEMM failed: " << int(status) << std::endl;
        return -1;
    }

    D.sync_host();
    std::cout << "GEMM + bias + GELU done." << std::endl;
}

