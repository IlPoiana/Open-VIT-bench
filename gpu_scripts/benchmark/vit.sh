# Kernel comparison
TOKENS_PER_BLOCK=(2) #keep only the best for the layer norm
ELEMENTS_PER_TH=(8)
TRANSPOSE_STRIDE=4
ADD_STRIDE=4

BATCH_N=(32)
BATCHES=(256)
MINIBATCHES=(128)
STREAMS_N=(2)

KERNEL_START=2 
KERNEL_END=5 

make clean 
for (( k=KERNEL_START; k<=KERNEL_END; k++)); do
    mkdir -p gpu_out/vit_bench/$k
    for b in "${BATCHES[@]}"; do
        for tok in "${TOKENS_PER_BLOCK[@]}"; do
            if [ "$tok" -lt "$b" ]; then
                make clean_bench_vit
                make test_bin/vit_bench.exe TOKENS_PER_BLOCK=$tok ELEMENTS_PER_TH=$ELEMENTS_PER_TH
                if [ "$k" -eq "2" ]; then
                    ./test_bin/vit_bench.exe --kernel $k --batch $b --transpose_stride $TRANSPOSE_STRIDE --add_stride $ADD_STRIDE > gpu_out/vit_bench/$k/$b.out
                else
                    for bn in "${BATCH_N[@]}"; do
                        if [ "$k" -lt "5" ]; then
                            ./test_bin/vit_bench.exe --kernel $k --batch $b --batch_n $bn --transpose_stride $TRANSPOSE_STRIDE --add_stride $ADD_STRIDE > gpu_out/vit_bench/$k/$b-$bn.out
                        else
                            make clean_gpu
                            make test_bin/vit_bench.exe MULTI_STREAM_WORKSPACE=1 TOKENS_PER_BLOCK=$tok ELEMENTS_PER_TH=$ELEMENTS_PER_TH
                            for mini in "${MINIBATCHES[@]}"; do
                                for stream in "${STREAMS_N[@]}"; do
                                    ./test_bin/vit_bench.exe --kernel $k --batch $b --batch_n $bn --minibatch $mini --streams_n $stream --transpose_stride $TRANSPOSE_STRIDE --add_stride $ADD_STRIDE > gpu_out/vit_bench/$k/$b-$bn-$mini.out
                                done
                            done
                        fi
                    done
                fi
            fi
        done
    done
done

echo finished