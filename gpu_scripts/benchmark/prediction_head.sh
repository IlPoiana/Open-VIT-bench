TOKENS_PER_BLOCK=(2 4) #keep only the best for the layer norm
ELEMENTS_PER_TH=(8)
# ELEMENTS_PER_TH=(1 2 4 8 16 32 64)
STRIDE=(4)
BLOCK_DIM=(256 512)

BATCHES=(2 4 8 16 32 64 128 256 512 1024 2048)
# BATCHES=(1 2 4 8)

mkdir -p gpu_out/prediction_head_bench
for tok in "${TOKENS_PER_BLOCK[@]}"; do
    make clean_bench_prediction_head
    make test_bin/prediction_head_bench.exe TOKENS_PER_BLOCK=$tok ELEMENTS_PER_TH=$ELEMENTS_PER_TH
    for b in "${BATCHES[@]}"; do
        for bdim in "${BLOCK_DIM[@]}"; do
            if [ "$tok" -lt "$((b + 1))" ]; then
                ./test_bin/prediction_head_bench.exe --kernel 1 --batch $b --stride $STRIDE --block_dim $bdim > gpu_out/prediction_head_bench/$b-$tok-$bdim.out
            fi
        done
    done
done

echo finished