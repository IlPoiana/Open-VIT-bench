CC := g++
CFLAGS := -std=c++11 -O3
OMPFLAGS := -fopenmp

BIN_FOLDER := bin
OBJ_FOLDER := obj
SRC_FOLDER := src

OMP_BIN_FOLDER := omp_bin
OMP_OBJ_FOLDER := omp_obj
OMP_SRC_FOLDER := omp_src

TEST_BIN_FOLDER := test_bin
TEST_OBJ_FOLDER := test_obj
TEST_SRC_FOLDER := test_src



all : vit

clean :
	rm -rf ./$(OBJ_FOLDER)/* ./$(BIN_FOLDER)/* ./$(OMP_OBJ_FOLDER)/* ./$(OMP_BIN_FOLDER)/* \
		   ./$(TEST_OBJ_FOLDER)/* ./$(TEST_BIN_FOLDER)/* \
		   ./out_comparison/* ./logs/*

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
\
$(OBJ_FOLDER)/datatypes.o \
$(OBJ_FOLDER)/modules.o \
$(OBJ_FOLDER)/mlp.o \
$(OBJ_FOLDER)/conv2d.o \
$(OBJ_FOLDER)/attention.o \
$(OBJ_FOLDER)/block.o \
$(OBJ_FOLDER)/patch_embed.o \
$(OBJ_FOLDER)/vision_transformer.o \
$(OBJ_FOLDER)/utils.o \
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
$(OBJ_FOLDER)/datatypes.o \
$(OBJ_FOLDER)/modules.o \
$(OBJ_FOLDER)/mlp.o \
$(OBJ_FOLDER)/conv2d.o \
$(OBJ_FOLDER)/attention.o \
$(OBJ_FOLDER)/block.o \
$(OBJ_FOLDER)/patch_embed.o \
$(OBJ_FOLDER)/vision_transformer.o \
$(OBJ_FOLDER)/utils.o \
$(TEST_OBJ_FOLDER)/%.o
	$(CC) $(CFLAGS) $^ -o $@
