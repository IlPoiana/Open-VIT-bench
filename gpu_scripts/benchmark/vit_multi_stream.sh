# Params search (MULTI STREAM)
TOKENS_PER_BLOCK=2 #keep only the best for the layer norm
ELEMENTS_PER_TH=8
TRANSPOSE_STRIDE=4
ADD_STRIDE=4

TOTAL_SAMPLES=(1024 2048 3072 4096 5120 6144 7168 8192)
#Test 1 - test what is the relation between minibatch size, streams_n
# BATCH_N=(4 8 16 32)
# STREAMS_N=(2 4 8 16)
BATCH_N=(16)
STREAMS_N=(16)

BATCH_MIN_DIM=8 # > 8 so 16
MINIBATCH_MAX_DIM=256 # < 256 so 128  
# Recover the Batch = 32 and minibatch = 128 

b=0
mini=0

export NVIDIA_TF32_OVERRIDE=0
make clean 
mkdir -p gpu_out/vit_bench/5
make test_bin/vit_bench.exe MULTI_STREAM_WORKSPACE=1 TOKENS_PER_BLOCK=$TOKENS_PER_BLOCK ELEMENTS_PER_TH=$ELEMENTS_PER_TH
for tot in "${TOTAL_SAMPLES[@]}"; do
    for bn in "${BATCH_N[@]}"; do
        b=$((tot / bn)) # BATCHES
        if [ "$b" -gt "$BATCH_MIN_DIM" ]; then # avoiding batches too small for many streams
            for stream in "${STREAMS_N[@]}"; do
                mini=$((b / stream)) # MINIBATCHES
                if [[ "$mini" -lt "$MINIBATCH_MAX_DIM" && "$mini" -gt "$((TOKENS_PER_BLOCK - 1))" ]]; then # avoiding to overflow the attention workspace (max 128 images per stream)
                    ./test_bin/vit_bench.exe --kernel 5 --batch $b --batch_n $bn --minibatch $mini --streams_n $stream --transpose_stride $TRANSPOSE_STRIDE --add_stride $ADD_STRIDE > gpu_out/vit_bench/5/$tot-$bn-$b-$mini.out
                fi
            done
        fi
    done
done

echo finished