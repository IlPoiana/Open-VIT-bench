SECTION=conv2d

make clean
make test_bin/bench_$SECTION.exe

# BATCHES=(1 2 4 8 16 32 64 128 256)
# IMG_SIZES=(64 128 224 512)
BATCHES=(1 2 4)
IMG_SIZES=(64 128 224)
LEVELS=2

for b in "${BATCHES[@]}"; do
    for size in "${IMG_SIZES[@]}"; do
        for l in $(seq 0 $LEVELS); do
            ./test_bin/bench_conv2d.exe data/$b-$size/pic_1.cpic models/vit_1.cvit $l 2 > json/blocks/$SECTION/$b-$size-$l.json
        done
    done
done