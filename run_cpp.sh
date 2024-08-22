source params.sh

if [ ! -d "out" ]; then
    mkdir "out"
fi
if [ ! -d "measures" ]; then
    mkdir "measures"
fi
if [ ! -d "out/cpp_1" ]; then
    mkdir "out/cpp_1"
fi
if [ ! -d "out/cpp_2" ]; then
    mkdir "out/cpp_2"
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

if [ ! -f "bin/vit.exe" ]; then
    echo Error: missing bin/vit.exe file!
    echo Run compile.sh script
    exit 1
fi



if [ ! -f "measures/cpp_1.csv" ]; then
    touch measures/cpp_1.csv
fi
echo "batch_size;model_depth;load_cvit_time;load_cpic_time;foreward_time[];store_cprd_time" >measures/cpp_1.csv
for i in $(seq 1 $DTASET_DIM); do
    ./bin/vit.exe models/vit_1.cvit data/pic_$i.cpic out/cpp_1/prd_$i.cprd measures/cpp_1.csv
done
echo first vit executed



if [ ! -f "measures/cpp_2.csv" ]; then
    touch measures/cpp_2.csv
fi
echo "batch_size;model_depth;load_cvit_time;load_cpic_time;foreward_time[];store_cprd_time" >measures/cpp_2.csv
for i in $(seq 1 $DTASET_DIM); do
    ./bin/vit.exe models/vit_2.cvit data/pic_$i.cpic out/cpp_2/prd_$i.cprd measures/cpp_2.csv
done
echo second vit executed
