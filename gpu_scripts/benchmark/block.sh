MLP_STRIDE=(4) #Fixed to one maybe two variables
BLOCK_DIM=(256)
TOKENS_PER_BLOCK=(4) #Keep only the good values from the tests
ELEMENTS_PER_TH=(8) #Keep only the good values from the tests

# BATCHES=(2 4 8 16 32 64 128 256 512 1024 2048)
BATCHES=(2048)

export NVIDIA_TF32_OVERRIDE=0
mkdir -p gpu_out/block_bench
for tok in "${TOKENS_PER_BLOCK[@]}"; do
    make clean_bench_block
    make test_bin/block_bench.exe TOKENS_PER_BLOCK=$tok ELEMENTS_PER_TH=$ELEMENTS_PER_TH
    for b in "${BATCHES[@]}"; do
        if [ "$tok" -lt "$b" ]; then
            ./test_bin/block_bench.exe --batch $b --kernel 3 --mlp_type 1 > gpu_out/block_bench/$b-$tok.out
        fi
    done
done

echo finished