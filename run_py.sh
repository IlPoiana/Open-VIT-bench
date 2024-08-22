source params.sh
#export CUDA_VISIBLE_DEVICES=""

if [ ! -d "out" ]; then
    mkdir "out"
fi
if [ ! -d "measures" ]; then
    mkdir "measures"
fi
if [ ! -d "out/py_1" ]; then
    mkdir "out/py_1"
fi
if [ ! -d "out/py_2" ]; then
    mkdir "out/py_2"
fi



if [ ! -d "models" ]; then
    echo Error: missing models folder!
    echo Run create_models.sh script
    exit 1
fi

if [ ! -d "data" ]; then
    echo Error: missing data folder!
    echo Run create_dataset.sh script
    exit 1
fi

if [ ! -f "timm_train_vit/vit.py" ]; then
    echo Error: missing timm_train_vit/vit.py file!
    exit 1
fi



if [ ! -f "measures/py_1.csv" ]; then
    touch measures/py_1.csv
fi
echo "batch_size;model_depth;load_cvit_time;load_cpic_time;foreward_time[];store_cprd_time" >measures/py_1.csv
for i in $(seq 1 $DTASET_DIM); do
    python3 timm_train_vit/vit.py models/vit_1.pt data/pic_$i.cpic out/py_1/prd_$i.cprd measures/py_1.csv
done
echo first vit executed



if [ ! -f "measures/py_2.csv" ]; then
    touch measures/py_2.csv
fi
echo "batch_size;model_depth;load_cvit_time;load_cpic_time;foreward_time[];store_cprd_time" >measures/py_2.csv
for i in $(seq 1 $DTASET_DIM); do
    python3 timm_train_vit/vit.py models/vit_2.pt data/pic_$i.cpic out/py_2/prd_$i.cprd measures/py_2.csv
done
echo second vit executed
