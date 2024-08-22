source params.sh

if [ ! -d "data" ]; then
    mkdir "data"
fi

for i in $(seq 1 $DTASET_DIM); do
    python3 scripts/random_cpic.py data/pic_$i.cpic $DTASET_MIN_B $DTASET_MAX_B $DTASET_C $DTASET_H $DTASET_W $DTASET_MIN_VAL $DTASET_MAX_VAL
done
echo datased created

if [ ! -d "logs" ]; then
    mkdir "logs"
fi
if [ ! -f "logs/dataset_info.txt" ]; then
    touch logs/dataset_info.txt
fi
echo "dataset dimension: $DTASET_DIM batches" >>logs/dataset_info.txt
echo "minimum batch size: $DTASET_MIN_B pictures" >>logs/dataset_info.txt
echo "maximum batch size: $DTASET_MAX_B pictures" >>logs/dataset_info.txt
echo "channel dimension: $DTASET_C" >>logs/dataset_info.txt
echo "picture height: $DTASET_H" >>logs/dataset_info.txt
echo "picture width: $DTASET_W" >>logs/dataset_info.txt
echo "pixel minimum value: $DTASET_MIN_VAL" >>logs/dataset_info.txt
echo "pixel maximum value: $DTASET_MAX_VAL" >>logs/dataset_info.txt

echo dataset info printed on file logs/dataset_info.txt
