source params.sh

if [ ! -d "out" ]; then
    mkdir "out"
fi
if [ ! -d "measures" ]; then
    mkdir "measures"
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

if [ ! -f "omp_bin/vit.exe" ]; then
    echo Error: missing omp_bin/vit.exe file!
    echo Run compile.sh script
    exit 1
fi



for num_threads in ${THREAD_LIST[@]}; do
    export OMP_NUM_THREADS=$num_threads



    if [ ! -d "out/omp_1_$num_threads" ]; then
        mkdir "out/omp_1_$num_threads"
    fi
    if [ ! -d "out/omp_2_$num_threads" ]; then
        mkdir "out/omp_2_$num_threads"
    fi



    if [ ! -f "measures/omp_1_$num_threads.csv" ]; then
        touch measures/omp_1_$num_threads.csv
    fi
    echo "batch_size;model_depth;load_cvit_time;load_cpic_time;foreward_time[];store_cprd_time" >measures/omp_1_$num_threads.csv
    for i in $(seq 1 $DTASET_DIM); do
        ./omp_bin/vit.exe models/vit_1.cvit data/pic_$i.cpic out/omp_1_$num_threads/prd_$i.cprd measures/omp_1_$num_threads.csv
    done
    echo first vit executed with $num_threads threads



    if [ ! -f "measures/omp_2_$num_threads.csv" ]; then
        touch measures/omp_2_$num_threads.csv
    fi
    echo "batch_size;model_depth;load_cvit_time;load_cpic_time;foreward_time[];store_cprd_time" >measures/omp_2_$num_threads.csv
    for i in $(seq 1 $DTASET_DIM); do
        ./omp_bin/vit.exe models/vit_2.cvit data/pic_$i.cpic out/omp_2_$num_threads/prd_$i.cprd measures/omp_2_$num_threads.csv
    done
    echo second vit executed with $num_threads threads
done
