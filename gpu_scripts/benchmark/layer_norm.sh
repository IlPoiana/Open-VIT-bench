# Kernel comparison
TOKENS_PER_BLOCK=(4)
ELEMENTS_PER_TH=(4)
BATCHES=(512)
KERNEL_START=1 
KERNEL_END=5 

# Parameters search
# TOKENS_PER_BLOCK=(2 4 8 16 32 64)
# ELEMENTS_PER_TH=(2 4 8 16 32 64)
# KERNEL_START=5 
# KERNEL_END=5 
# BATCHES=(2 4 8 16 32 64 128 256 512)

for (( k=KERNEL_START; k<=KERNEL_END; k++)); do
    mkdir -p gpu_out/layer_norm_bench/$k
    for b in "${BATCHES[@]}"; do
        for tok in "${TOKENS_PER_BLOCK[@]}"; do
            for elem in "${ELEMENTS_PER_TH[@]}"; do
                if [ "$tok" -lt "$b" ]; then
                    if [ "$k" -eq "5" ]; then
                        make clean_bench_layer
                        make test_bin/layer_norm_bench.exe TOKENS_PER_BLOCK=$tok ELEMENTS_PER_TH=$elem
                    fi 
                    ./test_bin/layer_norm_bench.exe --kernel $k --batch $b --tokens_per_block $tok > gpu_out/layer_norm_bench/$k/$b-$tok-$elem.out
                fi
            done
        done
    done
done

echo finished