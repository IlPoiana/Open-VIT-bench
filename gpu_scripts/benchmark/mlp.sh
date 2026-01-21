# STRIDE=(1 2 4 8 16 32 64 128)
# BLOCK_DIM=(64 128 256 512)
STRIDE=(2 4 8 16 32 64 128)
BLOCK_DIM=(64 256 512)
BATCHES=(2 4 8 16 32 64 128 256 512 1024 2048)
# BATCHES=(1 4 8)

make test_bin/mlp_bench.exe
mkdir -p gpu_out/mlp_bench/$k
for b in "${BATCHES[@]}"; do
    for stride in "${STRIDE[@]}"; do
        for bdim in "${BLOCK_DIM[@]}"; do
            ./test_bin/mlp_bench.exe --kernel 1 --batch $b --stride $stride --block_dim $bdim > gpu_out/mlp_bench/1/$b-$stride-$bdim.out
        done
    done
    ./test_bin/mlp_bench.exe --kernel 2 --batch $b > gpu_out/mlp_bench/2/$b.out
done



echo finished