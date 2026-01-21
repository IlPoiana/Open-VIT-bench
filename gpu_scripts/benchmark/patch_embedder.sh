# BLOCK_DIM=(64 128 256 512)
# TRANSPOSE_STRIDE=(1 2 4 8 16 32 64)
# POS_STRIDE=(1 2 4 8 16 32 64)
BLOCK_DIM=(128 256 512)
TRANSPOSE_STRIDE=(2 4 8 16 32 64)
POS_STRIDE=(2 4 8 16 32 64)


# BATCHES=(2 4 8 16 32 64 128 256 512 1024)
BATCHES=(1024 2048)

# BATCHES=(2 4 8)

mkdir -p gpu_out/patch_embedder_bench
make test_bin/patch_embed_bench.exe
for b in "${BATCHES[@]}"; do
    for block in "${BLOCK_DIM[@]}"; do
        for trans in "${TRANSPOSE_STRIDE[@]}"; do
            for pos in "${POS_STRIDE[@]}"; do
                ./test_bin/patch_embed_bench.exe --kernel 1 --batch $b --block_dim $block --transpose_stride $trans --pos_emb_stride $pos > gpu_out/patch_embedder_bench/$b-$block-$trans-$pos.out
            done
        done
    done
done

echo finished