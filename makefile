CC := g++
CFLAGS := -std=c++11 -O3
OMPFLAGS := -fopenmp

BIN_FOLDER := bin
OBJ_FOLDER := obj
SRC_FOLDER := src
GPU_SRC_FOLDER := gpu_src

OMP_BIN_FOLDER := omp_bin
OMP_OBJ_FOLDER := omp_obj
OMP_SRC_FOLDER := omp_src

TEST_BIN_FOLDER := test_bin
TEST_OBJ_FOLDER := test_obj
TEST_SRC_FOLDER := test_src

CPU_COMMON = \
	$(OBJ_FOLDER)/datatypes.o \
	$(OBJ_FOLDER)/modules.o \
	$(OBJ_FOLDER)/mlp.o \
	$(OBJ_FOLDER)/conv2d.o \
	$(OBJ_FOLDER)/attention.o \
	$(OBJ_FOLDER)/block.o \
	$(OBJ_FOLDER)/patch_embed.o \
	$(OBJ_FOLDER)/vision_transformer.o \
	$(OBJ_FOLDER)/utils.o

GPU_BENCH_FOLDER := gpu_benchmark

ARCH ?= -arch=sm_80
CUDA_FLAGS := $(ARCH) -lcublas -lcublasLt
CUDNN_FE := -I/home/emanuele.poiana/Open-VIT-bench/cudnn
CUDNN_FLAGS := -lcuda -lcudnn

all : vit

clean :
	rm -rf ./$(OBJ_FOLDER)/* ./$(BIN_FOLDER)/* ./$(OMP_OBJ_FOLDER)/* ./$(OMP_BIN_FOLDER)/* \
		   ./$(TEST_OBJ_FOLDER)/* ./$(TEST_BIN_FOLDER)/* \
		   ./out_comparison/* ./logs/*

clean_data :
	rm -rf ./data/*

clean_json :
	rm -rf ./json/blocks/conv2d/* ./json/vit/*

clean_gpu_out :
	rm -rf ./gpu_out/*

clean_everything :
	rm -rf ./$(OBJ_FOLDER)/* ./$(BIN_FOLDER)/* ./$(OMP_OBJ_FOLDER)/* ./$(OMP_BIN_FOLDER)/* \
		   ./$(TEST_OBJ_FOLDER)/* ./$(TEST_BIN_FOLDER)/* ./test_files/* \
		   ./data/* ./models/* ./out/* ./measures/* \
		   ./out_comparison/* ./logs/*

vit : % : $(BIN_FOLDER)/%.exe



# OBJs
$(OBJ_FOLDER)/datatypes.o \
$(OBJ_FOLDER)/modules.o \
$(OBJ_FOLDER)/mlp.o \
$(OBJ_FOLDER)/conv2d.o \
$(OBJ_FOLDER)/attention.o \
$(OBJ_FOLDER)/block.o \
$(OBJ_FOLDER)/patch_embed.o \
$(OBJ_FOLDER)/vision_transformer.o \
$(OBJ_FOLDER)/utils.o \
$(OBJ_FOLDER)/main.o \
\
: $(OBJ_FOLDER)/%.o : $(SRC_FOLDER)/%.cpp
	$(CC) -c $(CFLAGS) $^ -o $@

# Executables
$(BIN_FOLDER)/vit.exe : \
$(CPU_COMMON) \
$(OBJ_FOLDER)/main.o
	$(CC) $(CFLAGS) $^ -o $@



# OMP OBJs
$(OMP_OBJ_FOLDER)/datatypes.o \
$(OMP_OBJ_FOLDER)/modules.o \
$(OMP_OBJ_FOLDER)/conv2d.o \
$(OMP_OBJ_FOLDER)/attention.o \
$(OMP_OBJ_FOLDER)/vision_transformer.o \
\
: $(OMP_OBJ_FOLDER)/%.o : $(OMP_SRC_FOLDER)/%.cpp
	$(CC) -c $(CFLAGS) $(OMPFLAGS) $^ -o $@

# OMP Executables
$(OMP_BIN_FOLDER)/vit.exe : \
\
$(OMP_OBJ_FOLDER)/datatypes.o \
$(OMP_OBJ_FOLDER)/modules.o \
$(OBJ_FOLDER)/mlp.o \
$(OMP_OBJ_FOLDER)/conv2d.o \
$(OMP_OBJ_FOLDER)/attention.o \
$(OBJ_FOLDER)/block.o \
$(OBJ_FOLDER)/patch_embed.o \
$(OMP_OBJ_FOLDER)/vision_transformer.o \
$(OBJ_FOLDER)/utils.o \
$(OBJ_FOLDER)/main.o
	$(CC) $(CFLAGS) $(OMPFLAGS) $^ -o $@



# Test OBJs
$(TEST_OBJ_FOLDER)/test_datatypes.o \
$(TEST_OBJ_FOLDER)/test_modules.o \
$(TEST_OBJ_FOLDER)/test_mlp.o \
$(TEST_OBJ_FOLDER)/test_conv2d.o \
$(TEST_OBJ_FOLDER)/test_attention.o \
$(TEST_OBJ_FOLDER)/test_block.o \
$(TEST_OBJ_FOLDER)/test_patch_embed.o \
$(TEST_OBJ_FOLDER)/test_vision_transformer.o \
$(TEST_OBJ_FOLDER)/test_utils.o \
\
: $(TEST_OBJ_FOLDER)/%.o : $(TEST_SRC_FOLDER)/%.cpp
	$(CC) -c $(CFLAGS) $^ -o $@

# Test Executables
$(TEST_BIN_FOLDER)/test_datatypes.exe \
$(TEST_BIN_FOLDER)/test_modules.exe \
$(TEST_BIN_FOLDER)/test_mlp.exe \
$(TEST_BIN_FOLDER)/test_conv2d.exe \
$(TEST_BIN_FOLDER)/test_attention.exe \
$(TEST_BIN_FOLDER)/test_block.exe \
$(TEST_BIN_FOLDER)/test_patch_embed.exe \
$(TEST_BIN_FOLDER)/test_vision_transformer.exe \
$(TEST_BIN_FOLDER)/test_utils.exe \
\
: $(TEST_BIN_FOLDER)/%.exe : \
\
$(CPU_COMMON)\
$(TEST_OBJ_FOLDER)/%.o
	$(CC) $(CFLAGS) $^ -o $@


########################## GPU IMPLEMENTATIONS 

# ALL OBJ TARGET
$(OBJ_FOLDER)/gpu_datatypes.o \
$(OBJ_FOLDER)/gpu_conv2d.o \
$(OBJ_FOLDER)/cuda_utils.o \
$(OBJ_FOLDER)/gpu_vit.o \
$(OBJ_FOLDER)/gpu_layer.o\
$(OBJ_FOLDER)/gpu_mlp.o \
$(OBJ_FOLDER)/cudnn_attention.o \
$(OBJ_FOLDER)/cudnn_conv2d.o \
$(OBJ_FOLDER)/gpu_block.o \
$(OBJ_FOLDER)/gpu_patch_embedder.o \
$(OBJ_FOLDER)/gpu_proj_head.o \
: $(OBJ_FOLDER)/%.o : $(GPU_SRC_FOLDER)/%.cu
	nvcc -c $(CUDA_FLAGS) $^ -o $@

# ATTENTION !!!
# COMPILING WITH sm=50, change ARCH for the cluster GPUs
$(OBJ_FOLDER)/cudnn_utils.o \
: $(OBJ_FOLDER)/%.o : $(GPU_SRC_FOLDER)/%.cu
	nvcc -c $(CUDA_FLAGS) $(CUDNN_FLAGS) $^ -o $@

# CUSTOM AND BENCH TARGET
# CUSTOM
obj/cuBLAS.o : $(CPU_COMMON)
	nvcc $(CUDA_FLAGS) $(GPU_SRC_FOLDER)/cuBLAS.cu $^ -o $(OBJ_FOLDER)/cuBLAS.o

obj/cuDNN_BLAS.o : $(CPU_COMMON) 
	nvcc $(CUDA_FLAGS) $(CUDNN_CLUSTER_FLAGS) $(GPU_SRC_FOLDER)/cuBLAS.cu $^ -o $(OBJ_FOLDER)/cuBLAS.o


obj/cudnn_backend_conv:
	nvcc -std=c++17 test_src/cuDNN_explained.cu -lcudnn -lcuda -o obj/cudnn_backend_conv

# CUDNN single components

$(TEST_OBJ_FOLDER)/test_gpu_mlp.o \
$(TEST_OBJ_FOLDER)/test_gpu_layer.o \
$(TEST_OBJ_FOLDER)/test_gpu_vit.o \
$(TEST_OBJ_FOLDER)/test_cudnn_attention.o \
$(TEST_OBJ_FOLDER)/test_cudnn_conv2d.o \
$(TEST_OBJ_FOLDER)/test_gpu_block.o \
$(TEST_OBJ_FOLDER)/test_gpu_patch_embed.o \
$(TEST_OBJ_FOLDER)/test_gpu_proj_head.o \
$(TEST_OBJ_FOLDER)/test_gpu_vit.o \
: $(TEST_OBJ_FOLDER)/%.o: $(TEST_SRC_FOLDER)/%.cu
	nvcc -c $(CUDA_FLAGS) $< -o $@

test_bin/test_cudnn_attention.exe \
test_bin/test_cudnn_conv2d.exe \
: $(TEST_BIN_FOLDER)/%.exe : \
$(CPU_COMMON) \
$(OBJ_FOLDER)/cuda_utils.o \
$(OBJ_FOLDER)/cudnn_utils.o \
$(OBJ_FOLDER)/cudnn_attention.o \
$(OBJ_FOLDER)/cudnn_conv2d.o \
$(OBJ_FOLDER)/gpu_datatypes.o \
$(TEST_OBJ_FOLDER)/%.o
	nvcc $(CUDA_FLAGS) $(CUDNN_FLAGS) $^ -o $@

# GPU single components

test_bin/test_gpu_layer.exe \
test_bin/test_gpu_mlp.exe \
: $(TEST_BIN_FOLDER)/%.exe : \
$(CPU_COMMON) \
$(OBJ_FOLDER)/cuda_utils.o \
$(OBJ_FOLDER)/gpu_datatypes.o \
$(OBJ_FOLDER)/gpu_layer.o \
$(OBJ_FOLDER)/gpu_vit.o \
$(OBJ_FOLDER)/gpu_mlp.o \
$(TEST_OBJ_FOLDER)/%.o
	nvcc $(CUDA_FLAGS) $^ -o $@

# MULTI-COMPONENTS
$(TEST_BIN_FOLDER)/test_gpu_patch_embed.exe \
: $(TEST_BIN_FOLDER)/%.exe : \
$(CPU_COMMON) \
$(OBJ_FOLDER)/cuda_utils.o \
$(OBJ_FOLDER)/gpu_datatypes.o \
$(OBJ_FOLDER)/gpu_layer.o \
$(OBJ_FOLDER)/cudnn_conv2d.o \
$(OBJ_FOLDER)/gpu_patch_embedder.o \
$(TEST_OBJ_FOLDER)/%.o
	nvcc $(CUDA_FLAGS) $(CUDNN_FLAGS) $^ -o $@


$(TEST_BIN_FOLDER)/test_gpu_block.exe \
: $(TEST_BIN_FOLDER)/%.exe : \
$(CPU_COMMON) \
$(OBJ_FOLDER)/cuda_utils.o \
$(OBJ_FOLDER)/gpu_datatypes.o \
$(OBJ_FOLDER)/gpu_layer.o \
$(OBJ_FOLDER)/gpu_mlp.o \
$(OBJ_FOLDER)/cudnn_attention.o \
$(OBJ_FOLDER)/cudnn_conv2d.o \
$(OBJ_FOLDER)/gpu_block.o \
$(TEST_OBJ_FOLDER)/%.o
	nvcc $(CUDA_FLAGS) $(CUDNN_FLAGS) $^ -o $@

test_bin/test_gpu_proj_head.exe \
: $(TEST_BIN_FOLDER)/%.exe : \
$(CPU_COMMON) \
$(OBJ_FOLDER)/cuda_utils.o \
$(OBJ_FOLDER)/gpu_datatypes.o \
$(OBJ_FOLDER)/gpu_layer.o \
$(OBJ_FOLDER)/gpu_mlp.o \
$(OBJ_FOLDER)/cudnn_attention.o \
$(OBJ_FOLDER)/gpu_proj_head.o \
$(TEST_OBJ_FOLDER)/%.o
	nvcc $(CUDA_FLAGS) $(CUDNN_FLAGS) $^ -o $@

test_bin/test_gpu_vit.exe \
: $(TEST_BIN_FOLDER)/%.exe : \
$(CPU_COMMON) \
$(OBJ_FOLDER)/cuda_utils.o \
$(OBJ_FOLDER)/gpu_datatypes.o \
$(OBJ_FOLDER)/cudnn_utils.o \
$(OBJ_FOLDER)/cudnn_attention.o \
$(OBJ_FOLDER)/cudnn_conv2d.o \
$(OBJ_FOLDER)/gpu_layer.o \
$(OBJ_FOLDER)/gpu_vit.o \
$(OBJ_FOLDER)/gpu_mlp.o \
$(OBJ_FOLDER)/gpu_patch_embedder.o \
$(OBJ_FOLDER)/gpu_block.o \
$(OBJ_FOLDER)/gpu_proj_head.o \
$(TEST_OBJ_FOLDER)/%.o
	nvcc $(CUDA_FLAGS) $(CUDNN_FLAGS) $^ -o $@
