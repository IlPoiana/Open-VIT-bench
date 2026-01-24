# C-ViT

Building upon the [C-ViT project](https://github.com/HicrestLaboratory/Open-VIT-bench)
 , this extension aims to provide a GPU-accelerated version of the C++ and OMP impelmentation. C-ViT programs are present to ensure a numerically correct baseline comparison for every part of the model.
**After the C++ and OpenMP sections, which have been kept largely unchanged from the original, you will find the GPU implementation.** 

## Table of contents

- [C-ViT](#c-vit)
  - [Table of contents](#table-of-contents)
  - [Files in this repository(C-ViT)](#files-in-this-repositoryc-vit)
  - [How to run the tests](#how-to-run-the-tests)
  - [How to run the benchmark](#how-to-run-the-benchmark)
- [GPU acceleration](#gpu-acceleration)
  - [Requirements](#requirements)
  - [Installation Guide](#installation-guide)
  - [Setup](#setup)
  - [Repo Structure](#repo-structure)
    - [Utilities and libraries](#utilities-and-libraries)
    - [Convolutional layer (cuDNN)](#convolutional-layer-cudnn)
    - [Multi-Head Self Attention (cuDNN)](#multi-head-self-attention-cudnn)
    - [Layer Norm (CUDA)](#layer-norm-cuda)
    - [MLP (cuBLAS)](#mlp-cublas)
    - [Patch Embedder](#patch-embedder)
    - [Encoder Block](#encoder-block)
    - [Prediction Head](#prediction-head)
    - [ViT](#vit)
  - [Benchmarks](#benchmarks)
    - [Component numerical tests](#component-numerical-tests)
    - [Component benchmarks](#component-benchmarks)
    - [Results reproducibility](#results-reproducibility)
    - [Visualization](#visualization)
  - [Maintainers](#maintainers)
    - [Project supervisors](#project-supervisors)



## Files in this repository(C-ViT)

In this repository you will find the following files and folders:

- *include/*: the folder containing the headers for C++.
  - *attention.h*: defines the multi head attention component.
  - *block.h*: defines the attention encoder block.
  - *conv2d.h*: defines the convolution component.
  - *datatypes.h*: defines the datatypes use in this project. They are `RowVector`, `Matrix`, `Tensor` `PictureBatch` and `PredictionBatch`.
  - *mlp.h*: defines the Multi Layer Perceptron component.
  - *modules.h*: defines other basic components, namely `Linear`, `LayerNorm`, `LayerScale`, `Activation`. It also includes `ReLU`, `GELU` and `global_pool_nlc` functions.
  - *patch_embed.h*: defines the `PatchEmbed` component, the one responsible of image tokenization.
  - *utils.h*: declares input/output functions.
  - *vision_transformer.h*: defines `VisionTransformer`, the wrapper class that will be used as the model.
- *src/*: the folder with the C++ source codes.
  - *main.cpp*: the file containing the `main` function. It instantiate VisionTransformer class, performs the `forward` operation and measures the time it takes.
  - All the other files are the implementation of the header with the same name.
- *omp_src/*: the folder with the C++ codes parallelized with OpenMP.
  - *attention.cpp*: parallelizes the `multi_head_attention` function.
  - *conv2d.cpp*: parallelizes the `forward` pass of the convolution.
  - *datatypes.cpp*: parallelizes the basic datatype operations such as sum, construction and padding.
  - *modules.cpp*: parallelizes the matrix product in `Linear`, the `forward` pass of `LayerNorm` and the element access in `LayerScale` and `Activation`.
  - *vision_transformer.cpp*: parallelizes the `position_embed` phase of the model.
- *test_src/*: this folder contains the files to test that each C++ component provides the same result of the python implementation. The name of the files indicates which element it refers to.
- *scripts/*: contains some python functions useful in testing and benchmarking.
  - *analyze_time_measures.py*: it loads a `.csv` file containing the time measures and extracts some statistics from it.
  - *compare_cpred.py*: given two prediction files, it checks whether they're values are similar (meaning their difference is below a given threshold). Useful to ensore two models produce the same output.
  - *create_tensor.py*: creates a torch tensor of the given shape.
  - *random_cpic.py*: creates a random `PictureBatch`, in a `.cpic` file.
  - *summary_cpred_comparison.py*: the comparison files created by *compare_cpred.py* can be quite dispersive, this script cumulates their information.
- *timm_train_vit/*: contains the python implementation and test files.
  - *timm/*: this folder has been given me by the FBK. It contais the actual implementation of a ViT in python.
  - *train.py*: this file was made by FBK as well, and can be used to train the model.
  - *vit.py*: it's the equivalent of C++ main, as it instantiates the model, call it's `foreward` method and measures the time.
  - *convert_pt_cvit.py*: converts the `.pt` model storage format used by python into the `.cvit` format I used for the C++ code.
  - *create_model.py*: creates a ViT model, and stores its parameters in a `.cvit` file for C++ as well as a `.pt` file for python.
  - *cvit_utils.py*: this file contains the function I created to make python model similar to the C++. `plot_tensor` guarantee the models plot information the same way, while the `PredictionBatch` class gives the two models the same output format. The rest of the files contains input-output functions that allow python to  read the format I designed for C++.
  - *print_model.py*: given a model parameter file, this script extract the most important information from it. With it you can easily understand what kind of input your model needs.
  - All the files that begin with `test_` are meant to be used in collaboration with C++ test sources. see the "How to run the tests" section below.
- *params.sh*: a bash script that exports the variables used by other scripts.
- *compile.sh*: this scripts compiles the C++ and OMP programs in their respective binary folders.
- *create_dataset.sh*: runs many times the `random_cpic.py` script to create a random dataset.
- *create_models.sh*: it's a bash wrapper for the python function `create_model.py`.
- *run_cpp.sh*: executes the benchmark of the C++ code see the "How to run the benchmark" section below.
- *run_omp.sh*: executes the benchmark of the OMP code see the "How to run the benchmark" section below.
- *run_py.sh*: executes the benchmark of the python code see the "How to run the benchmark" section below.
- *elaborate.sh*: runs the necessary python scripts to analyzed the data produced by the benchmark and puts the results in the `logs/` folder.
- *Makefile*: the makefile that contains the recipes to compile the source codes.

The execution of the programs will lead to the creation and filling of the following folders:

- *obj/*: intermediate folder containing object files for C++ code.
- *bin/*: the folder that contains the C++ executable.
- *omp_obj/*: intermediate folder containing object files for OMP code.
- *omp_bin/*: the folder that contains the OMP executable.
- *test_obj/*: intermediate folder containing object files of the test codes.
- *test_bin/*: the folder that contains the executables of the test codes.
- *test_files/*: the folder that contains the files processed by the test codes.
- *data/*: this is the place where the input data will be stored.
- *models/*: the ViT model parameters will be stored here.
- *out/*: here is where the programs will put their results.
- *measures/*: the place where time measurements are stored.
- *out_comparison/*: in this folder will be put the comparison files generated by *compare_cpred.py* script.
- *logs/*: it's the place where you can find the final statistics of the benchmark, regarding the dataset, the model, the times and the outputs.

## How to run the tests

> [!Warning]
> Is necessary to create `obj/`, `test_obj/`, `test_bin/` to properly work.

The test files in the *test_src/* and *timm_train_vit/* are meant to show that the C++ and the python code produce the same results. I designed the tests to be executed in two parallel terminals.

In the first terminal, go to the *timm_train_vit/* and execute `python3 test_<component_name>.py`. On the other terminal first compile the C++ code with `make test_bin/test_<component_name>.exe`, then execute it with `./test_bin/test_<component_name>.exe`. You will appreciate the same result in both the terminals, meaning the floating point numbers will differ only in the less significant decimal digits.

Each test files simply instantiate a component, sets its parameters, creates an input and feed it in the forward function. If you wish, you can manually change the inputs and parameters acting in the source codes. The *create_tensor.py* scripts can help you: executing `python3 create_tensor.py <shape>` you will see printed the python code to instantiate a tensor fill with random numbers, and the C++ code to do the same. This ensure the two files will work on exactly the same data.

## How to run the benchmark

> [!Tip]
> We suggest to create your own *venv* to be able to perform all the python scripts. [Installation Guide](#installation-guide)

The benchmark is intended to compare the performance of the different ViT implementation, as well as show that the models produce the same output. The first step is the compilation of C++ and OMP in the *bin/* and *omp_bin/* respectively. You can do it running `bash compile.sh`.

Then, run `bash create_dataset.sh` to create a random dataset in the *data/* folder. You can adjust the dataset creation parameters in the appropriate section of *params.sh* file. Continue running `bash create_models.sh`: it will generate the ViT models and store them in the *models/* folder.

The core of the benchmark is represented by `bash run_cpp.sh`, `bash run_py.sh` `bash run_omp.sh`. These scripts execute the respective models, storing their output in the *out/* folder and their time measures in the *measures/* folder. You can control the number of OMP threads acting on *params.sh*.

Finally, execute `bash elaborate.sh`, it will compare each C++ output with the correspondent output of the other two models ensuring that all of them behave the same. Each comparison will be stored in the *out_comparison/* folder, but they will also be collapsed in a single file in the *logs/* folder. This same script will analyze time measures as well, and the result it gets will again be put in the *logs/* folder.

At the end of the benchmark, you will have everything you need in the *logs/* folder:

- *dataset_info.txt*: Contains the parameters used to create the dataset.
- *model_info.txt*: Contains the main attributes of the models used.
- *output_analysis.txt*: Here you can find the comparison pf the outputs. Use this file to understand if the models produce the same predictions or behave differently.
- *measures_analysis.txt*: This file contains some time statistics for each model. You can use it to understand which model performs better.

When you want to clean the work space, you have two possible commands:

- `make clean`: removes the bin folders, the obj folders, *out_comparison/* and *logs/*, but it leaves the dataset and the models untouched. It is used to perform different benchmarks on the same dataset-model pair.
- `make clean_everything`: removes all the generated folder, leaving the folder as just cloned. It is used when you also want a new dataset and a new model for your next benchmark.



---

# GPU acceleration
In this section are briefly explained how are made and how to use the GPU accelerated components.

## Requirements
- CUDA Development Toolkit 12.1.1, in particular cuBLASLt and CUB
- cuDNN 8.9.2 (Backend only, not frontend)
- Python 3.10 or more & the packages included in requirements.txt


## Installation Guide
Here some information about CUDA development toolkit and cuDNN for setting up.

- [CUDA dev toolkit 12.1.1](https://developer.nvidia.com/cuda-12-1-1-download-archive)
- [CUDA doc for 12.1.1](https://docs.nvidia.com/cuda/archive/12.1.1/)
- [cuDNN 8.9.2 Installation guide](https://docs.nvidia.com/deeplearning/cudnn/archives/cudnn-892/install-guide/index.html)
- [cuDNN 8.9.2 doc](https://docs.nvidia.com/deeplearning/cudnn/archives/cudnn-892/api/index.html#overview)



**Python dependecies**

Python scripts are used for data and model generation in `test_gpu_vit.exe`, other than benchmark data extraction and visualization.
> [!Note]
> **Python is not strictly required** for running all the **benchmark scripts**, the **individual benchmarks** and the `test_gpu_*` files except for `test_gpu_vit.exe` 

1. If you haven't already done, clone the repository:
```bash
git clone https://github.com/IlPoiana/Open-VIT-bench.git
cd Open-VIT-bench
```

2. Create a virtual environment (recommended):
```bash
python -m venv gpu-vit
source gpu-vit/bin/activate
```

3. Install the packages:
```bash
pip install -r requirements.txt
```


## Setup
1. Create all the work directories.

```bash
mkdir -p obj test_obj test_bin gpu_out
```

## Repo Structure
- *gpu_include:* this folder contains all the header files of all the gpu components and utility programs. Follows the same logic as the cpp section.
- *gpu_src:* this folder contains all the source files with the C++ & CUDA implementations following the same logic as the cpp section.
- *gpu_benchmark:* this folder contains each component time benchmarks.     
- *gpu_scripts:* this folder contains a series of bash and python scripts used through the project for running benchmarks and extracting and visualizing data from them. 
- *gpu_out:* All the output files generated by the benchmarks are saved here. So any bash script under the `gpu_benchmark/benchmark/` directory.
- *test_src:* in the same folder as the previous section there are the source code for the test implementations.
- *test_obj:* like the CPU version, all the unlinked object files from `test_bin/test_*.exe` or `test_bin/*_bench.exe` made through the corresponding make command, will be saved here.
- *test_bin:* like the CPU version, all the executable files compiled and linked, will be stored in this directory.
- *obj:* like the CPU version, all the unlinked object files will be saved in this directory when compiling through the `make` command.

Following, a brief description of how each component is handled in this project.
### Utilities and libraries
Each header file depedends on the precedent. Are described in a bottom up order (from the most atomic to the one that includes all the others).

- `cuda_utils`: define some important shared macros, and common use functions tied to kernels routines or operations

> [!Note]
> `helpers.h` imported function was taken from the cuDNN front-end during dev phase, only one or two functions are actually used. 

- `gpu_datatypes`: define some useful data structure, mainly used in dev phase and only in some `test_gpu_*` components. It imports the cuBLASLt library separately from `cuda_utils` to have better control on the compilation of single components that doesn't require cuBLASLt.

- `cudnn_utils`: used mainly during dev phase where different cudnn inherent functions were defined, now deprecated and removed, it imports the cuDNN library separately from `cuda_utils` to have better control on the compilation of single components that doesn't require cuDNN. 

- `bench_utils`: define all the data structure and methods used for benchmarking the kernels

### Convolutional layer (cuDNN) 
- `cudnn_conv2d`: define all the methods and data structures for the convolutional layer. Following the most relevant:
  - `init_conv2d_descriptors`: initialize the cuDNN descriptors for the conv2d op and saves them in the *convolution_desc* data structure.
  - `execute_cudnn_conv2d_bias`: execute the conv2d with the descriptors initialized with the precedent method.
### Multi-Head Self Attention (cuDNN)
- `cudnn_attention`: define all the methods and data structures for the multi-head attention operation. Following the most relevant:
  - `initialize_attn_descriptors`: initialize all the required cuDNN descriptors for the mha and save them in the *attn_cuDNN_descriptors* data structure.
  - `allocate_attn_weights`: uses cuDNN methods for fetching expected weights and workspace buffer sizes and allocate them.
  - `load_attn_weights`: uses cuDNN methods for load the mha weights where cuDNN expected to be(for more info refer to [cuDNN] doc).
  - `attention_device`: executes the mha through cuDNN.

### Layer Norm (CUDA)
- `gpu_layer`: define all the kernels and macros for the layer norm operation (ViT layer norm). Each kernel represent the same operation introducing over and over improvements (and sometimes losing flexibility) to it. Following the most relevant:
  - `gpu_layer_norm`: First version, without CUB, shows some of the main concepts 
  - `cub_layer_norm`: Like the first version, but with CUB for reduction and multiple tokens per block.
  - `multi_elem_cub_ln`: Like `cub_layer_norm` but with multiple elements per thread (more than 2)
  - `unrolled_multi_elem_cub_ln`: the same as `multi_elem_cub_ln` but with loop unrolling.

### MLP (cuBLAS)
- `gpu_mlp`: 
  - `create_mlp_descriptors`: initialized the cuBLAS descriptors and others data structure in *cublasLt_matmul_desc* data structure. 
  - `gpu_mlp(method)`: an mlp implementation with cuBLASLt and a bias add kernel made by me
  - `fused_gpu_mlp`: an mlp implementation with cuBLASLt only. the bias add and GELU are fused in the epilogue.
---

> [!Note]
> All the composed classes have implemented some common methods, like `forward`, `init_descriptors`, `destroy_descriptors` and more. I'm not focusing on the details of each but essentially they do a certain operation specifically for that component.

**Most important common methods found in different composed components:**
1. `init_descriptors`: initialize library specific descriptors.
2. `allocate/set_buffers`: allocate on device the buffers for storing results and intermediate operations. Set them for reusing in a ViT instance across different components.
3. `allocate/set_weights`: allocate on device the weights buffers for model weights. Set them for reusing in different ViT instances across multi stream implementation.
4. `load_weights`: Asyncronously load of model weights on previously allocated device buffers.
5. `forward`: Execute the forward function of that model component.
6. `destroy_descriptors`: destroy library specific descriptors.
7. `free buffers/weights`: handle the shared weights free and deallocation. 
---

### Patch Embedder
- `gpu_patch_embedder`: is the class representing the patch embedder + positional embeddings. It incorporates `cudnn_conv2d`.

### Encoder Block
- `gpu_block`: is the class representing the encoder block (inference only). Composed by `gpu_layer`, `gpu_mlp`, and `cudnn_attention`.

### Prediction Head
- `gpu_pred_head`: is the class representing the prediction head. Composed by layer_norm, a linear layer and softmax.

### ViT
- `gpu_vit`: is the class representing the vit model and implementations. Composed by the precedent components, `gpu_patch_embedder` `gpu_block` `gpu_pred_head`. 

## Benchmarks
### Component numerical tests  
These tests are designed for verifying the correctness of each routine, taking the CPU implementation as reference.

Each `test_gpu_*` program generates a random input and weights sets, shares them between the CPU and GPU implementation, and compute MRE between them.

Example
```bash
make test_bin/test_gpu_layer.exe
test_bin/test_gpu_layer.exe
```


**gpu_vit test | Datasets creations and testing**

`test_gpu_vit` has a different approach for testing it. It is possible, utilizing the procedure described [here](#how-to-run-the-benchmark), to load a vit model, translated from pytorch and convert it in C-ViT then in GPUVit and test it.

When the model is created and stored, is possible to create a dataset, it's important to use the image size as [224,224] and the number of channels equal to 3 cause the model have been made almost identical to the original ViT and the GPU implementation has strict requirements in the actual version.  

```bash
test_bin/test_gpu_vit.exe data/pic_1.cpic
```
### Component benchmarks 
This benchmarks are designed for testing each component performance. 
All the bechmarks are configurable by passing the corresponing flag to them. Check the respective file `main` source code under the `gpu_benchmark/` directory for all the possible parameters.


```bash
make test_bin/<component_name>_bench.exe --flag1 val1 --flag2 val2
```

Each output will have in the end a json-like object which represents the important data saved from the executed benchmark run.

> [!Warning]
> The `--cpu 1 ` flag is present in every bench file, it computes a baseline for numerical checking as CPU version. I strongly suggest **to not use it for large batches** due to the large time involved in the process. For simple numerical checks there are also the `test_gpu_*` programs

> [!Warning]
> For the **layer norm and all the components that have it inside**, if you want to change the default value, is necessary to compile the file through the `TOKENS_PER_BLOCK` and `ELEMENTS_PER_TH` flags. This can be took as reference from the bash benchmarks files and an example is given down below.

> [!Warning]
> Also the `MULTI_STREAM_WORKSPACE` flag is responsible for deciding **how much space allocate as workspace for the mha and mlp operations.**
> 
> 0 is the standard value and are 8GB(yes GB) and 1 is for multi-stream vit and is 512MB. If you are using small batches is safe to compile with value 1 to save on memory (except on multi stream that can catch up on 8GB with 16 streams)
>
> It's value can be found and set on `cuda_utils.h`, where all the files import it.


Examples 
```bash
# Patch Embedder 
./test_bin/patch_embed_bench.exe --batch 2 --cpu 1 
./test_bin/patch_embed_bench.exe --batch 256 
```

```bash
# Layer Norm
make test_bin/layer_norm_bench.exe TOKENS_PER_BLOCK=8 ELEMENTS_PER_TH=4
./test_bin/layer_norm_bench.exe --kernel 5 --batch 128

./test_bin/layer_norm_bench.exe --kernel 4 --batch 128 --tokens_per_block 8
```

```bash 
# Vit
./test_bin/vit_bench.exe --kernel 2 --batch 128 --transpose_stride 2 --add_stride 32

# kernel 5 is the multi stream, so more parameters are taken in consideration
./test_bin/vit_bench.exe --kernel 5 --batch_n 4 --batch 64  --minibatch 8 --streams_n 8 --transpose_stride 2 --add_stride 2

```

### Results reproducibility
To reproduce the results, I suggest to use the bash scripts present under `gpu_scripts/benchmarks` directory.
Alternatively, is possible to launch singularly benchmarks through the `*_bench.cu` programs present under the `benchmarks`

Example
```bash
bash gpu_scripts/benchmarks/vit.sh
```

All the tests have been executed on an NVIDIA A30 GPU, hosted on the University Of Trento "baldo" cluster.

### Visualization
python scripts

TO DO
## Maintainers

- *Alex Pegoraro* - [CPU](#c-vit) baseline - [GitHub](https://github.com/AlphaNightLight)
- *Emanuele Poiana* - [GPU](#gpu-acceleration) extension - [GitHub](https://github.com/IlPoiana)

[cuDNN]: https://docs.nvidia.com/deeplearning/cudnn/archives/cudnn-892/api/index.html#overview

### Project supervisors
- Flavio Vella
- Lorenzo Picchetti