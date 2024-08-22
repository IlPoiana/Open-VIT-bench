# Dataset parameters
export DTASET_DIM="8"
export DTASET_MIN_B="4"
export DTASET_MAX_B="16"
export DTASET_C="3"
export DTASET_H="224"
export DTASET_W="224"
export DTASET_MIN_VAL="0.0"
export DTASET_MAX_VAL="1.0"

# Prediction comparison parameters
export CPRD_HIGH_THRESHOLD="0.0001" # equal to 0.01 %
export CPRD_LOW_THRESHOLD="0.000001"

# OMP threads parameter
export THREAD_LIST=(1 2 4 8 16)
