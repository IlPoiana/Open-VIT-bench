source params.sh

if [ ! -d "out_comparison" ]; then
    mkdir "out_comparison"
fi
if [ ! -d "out_comparison/vit_1" ]; then
    mkdir "out_comparison/vit_1"
fi
if [ ! -d "out_comparison/vit_2" ]; then
    mkdir "out_comparison/vit_2"
fi
if [ ! -d "logs" ]; then
    mkdir "logs"
fi

if [ ! -f "scripts/compare_cpred.py" ]; then
    echo Error: missing scripts/compare_cpred.py file!
    exit 1
fi



# First Model Comparison
if [ ! -d "out/cpp_1" ]; then
    echo Error: missing out/cpp_1 folder!
    echo Run run_cpp.sh script
    exit 1
fi

if [ ! -d "out/py_1" ]; then
    echo Error: missing out/py_1 folder!
    echo Run run_py.sh script
    exit 1
fi

for i in $(seq 1 $DTASET_DIM); do
    python3 scripts/compare_cpred.py out/cpp_1/prd_$i.cprd out/py_1/prd_$i.cprd out_comparison/vit_1/cpp_vs_py.txt $CPRD_HIGH_THRESHOLD $CPRD_LOW_THRESHOLD
done
echo cpp and py compared for first model

for num_threads in ${THREAD_LIST[@]}; do
    if [ ! -d "out/omp_1_$num_threads" ]; then
        echo Error: missing out/omp_1_$num_threads folder!
        echo Run run_omp.sh script
        exit 1
    fi
    for i in $(seq 1 $DTASET_DIM); do
        python3 scripts/compare_cpred.py out/cpp_1/prd_$i.cprd out/omp_1_$num_threads/prd_$i.cprd out_comparison/vit_1/cpp_vs_omp_$num_threads.txt $CPRD_HIGH_THRESHOLD $CPRD_LOW_THRESHOLD
    done
    echo cpp and omp_$num_threads compared for first model
done



# Second Model Comparison
if [ ! -d "out/cpp_2" ]; then
    echo Error: missing out/cpp_2 folder!
    echo Run run_cpp.sh script
    exit 1
fi

if [ ! -d "out/py_2" ]; then
    echo Error: missing out/py_2 folder!
    echo Run run_py.sh script
    exit 1
fi

for i in $(seq 1 $DTASET_DIM); do
    python3 scripts/compare_cpred.py out/cpp_2/prd_$i.cprd out/py_2/prd_$i.cprd out_comparison/vit_2/cpp_vs_py.txt $CPRD_HIGH_THRESHOLD $CPRD_LOW_THRESHOLD
done
echo cpp and py compared for second model



for num_threads in ${THREAD_LIST[@]}; do
    if [ ! -d "out/omp_2_$num_threads" ]; then
        echo Error: missing out/omp_2_$num_threads folder!
        echo Run run_omp.sh script
        exit 1
    fi
    for i in $(seq 1 $DTASET_DIM); do
        python3 scripts/compare_cpred.py out/cpp_2/prd_$i.cprd out/omp_2_$num_threads/prd_$i.cprd out_comparison/vit_2/cpp_vs_omp_$num_threads.txt $CPRD_HIGH_THRESHOLD $CPRD_LOW_THRESHOLD
    done
    echo cpp and omp_$num_threads compared for second model
done



# Output Comparison Summary
if [ ! -f "scripts/summary_cpred_comparison.py" ]; then
    echo Error: missing scripts/summary_cpred_comparison.py file!
    exit 1
fi
if [ ! -f "logs/output_analysis.txt" ]; then
    touch logs/output_analysis.txt
fi
echo "" >logs/output_analysis.txt

python3 scripts/summary_cpred_comparison.py out_comparison/vit_1/cpp_vs_py.txt logs/output_analysis.txt
for num_threads in ${THREAD_LIST[@]}; do
    python3 scripts/summary_cpred_comparison.py out_comparison/vit_1/cpp_vs_omp_$num_threads.txt logs/output_analysis.txt
done

python3 scripts/summary_cpred_comparison.py out_comparison/vit_2/cpp_vs_py.txt logs/output_analysis.txt
for num_threads in ${THREAD_LIST[@]}; do
    python3 scripts/summary_cpred_comparison.py out_comparison/vit_2/cpp_vs_omp_$num_threads.txt logs/output_analysis.txt
done
echo comparisons analyzed on file logs/output_analysis.txt



# Measure Analysis
if [ ! -f "logs/measures_analysis.txt" ]; then
    touch logs/measures_analysis.txt
fi
echo "" >logs/measures_analysis.txt

python3 scripts/analyze_time_measures.py measures/cpp_1.csv logs/measures_analysis.txt
python3 scripts/analyze_time_measures.py measures/py_1.csv logs/measures_analysis.txt
for num_threads in ${THREAD_LIST[@]}; do
    python3 scripts/analyze_time_measures.py measures/omp_1_$num_threads.csv logs/measures_analysis.txt
done

python3 scripts/analyze_time_measures.py measures/cpp_2.csv logs/measures_analysis.txt
python3 scripts/analyze_time_measures.py measures/py_2.csv logs/measures_analysis.txt
for num_threads in ${THREAD_LIST[@]}; do
    python3 scripts/analyze_time_measures.py measures/omp_2_$num_threads.csv logs/measures_analysis.txt
done
echo measures analyzed on file logs/measures_analysis.txt
