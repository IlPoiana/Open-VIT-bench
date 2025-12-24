# Grid benchmark datasets creation
BATCHES=(1 2 4 8 16 32 64 128 256)
# IMG_SIZES=(64 128 224 512)
IMG_SIZES=(224)

# BATCHES=(1 2)
# IMG_SIZES=(64 128)

# Constant parameters
DTASET_C="3"
DTASET_MIN_VAL="0.0"
DTASET_MAX_VAL="1.0"
# Prediction comparison parameters
CPRD_HIGH_THRESHOLD="0.0001" # equal to 0.01 %
CPRD_LOW_THRESHOLD="0.000001"
# OMP threads parameter
THREAD_LIST=(1 2 4 8 16)

if [ ! -d "data" ]; then
    mkdir "data"
fi

# Dataset parameters
DTASET_DIM="1" #8

# iterate over BATCHES
for b in "${BATCHES[@]}"; do
    echo "Batch size: $b"
    DTASET_MIN_B="$b"
    DTASET_MAX_B="$b"
    echo DTASET_MIN_B $DTASET_MIN_B
    # iterate over IMG_SIZES
    for s in "${IMG_SIZES[@]}"; do
        echo "Image size: ${s}x${s}"
        DTASET_H="$s"
        DTASET_W="$s"
        echo DTASET_H $DTASET_H
        for i in $(seq 1 $DTASET_DIM); do
            if [ ! -d "data/$DTASET_MIN_B-$DTASET_H" ]; then
                mkdir "data/$DTASET_MIN_B-$DTASET_H"
            fi
            python3 scripts/random_cpic.py data/$DTASET_MIN_B-$DTASET_H/pic_$i.cpic $DTASET_MIN_B $DTASET_MAX_B $DTASET_C $DTASET_H $DTASET_W $DTASET_MIN_VAL $DTASET_MAX_VAL
        done
        echo datased $DTASET_MIN_B-$DTASET_H created

    done
done
#  DTASET_H="224"
#  DTASET_W="224"






# if [ ! -d "logs" ]; then
#     mkdir "logs"
# fi
# if [ ! -f "logs/dataset_info.txt" ]; then
#     touch logs/dataset_info.txt
# fi
# echo "dataset dimension: $DTASET_DIM batches" >>logs/dataset_info.txt
# echo "minimum batch size: $DTASET_MIN_B pictures" >>logs/dataset_info.txt
# echo "maximum batch size: $DTASET_MAX_B pictures" >>logs/dataset_info.txt
# echo "channel dimension: $DTASET_C" >>logs/dataset_info.txt
# echo "picture height: $DTASET_H" >>logs/dataset_info.txt
# echo "picture width: $DTASET_W" >>logs/dataset_info.txt
# echo "pixel minimum value: $DTASET_MIN_VAL" >>logs/dataset_info.txt
# echo "pixel maximum value: $DTASET_MAX_VAL" >>logs/dataset_info.txt

# echo dataset info printed on file logs/dataset_info.txt
