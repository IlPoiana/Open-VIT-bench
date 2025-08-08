# C-ViT

Starting from the C-ViT project, this extension aims to be implement a GPU accelerated benchmarking reference.
The project structure follows the original one to mantain all the original functions, blocks and tests accessible for comparison. After the cpp and omp sections(kept for the most same as the original) there is the GPU one.  

## Files in this repository

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
  - All the files that begin with *test_* are meant to be used in collaboration with C++ test sources. see the "How to run the tests" section below.
- *params.sh*: a bash script that exports the variables used by other scripts.
- *compile.sh*: this scripts compiles the C++ and OMP programs in their respective binary folders.
- *create_dataset.sh*: runs many times the *random_cpic.py* script to create a random dataset.
- *create_models.sh*: it's a bash wrapper for the python function *create_model.py*.
- *run_cpp.sh*: executes the benchmark of the C++ code see the "How to run the benchmark" section below.
- *run_omp.sh*: executes the benchmark of the OMP code see the "How to run the benchmark" section below.
- *run_py.sh*: executes the benchmark of the python code see the "How to run the benchmark" section below.
- *elaborate.sh*: runs the necessary python scripts to analyzed the data produced by the benchmark and puts the results in the *logs/* folder.
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

The benchmark is intended to compare the performance of the different ViT implementation, as well as show that the models produce the same output. The first step is the compilation of C++ and OMP in the *bin/* and *omp_bin/* respectively. You can do it running `bash compile.sh`.

Then, run `bash create_dataset.sh` to create a random dataset in the *data/* folder. You can adjust the dataset creation parameters in the appropriate section of *params.sh* file. Continue running `bash create_models.sh`: it will generate the ViT models and store them in the *models/* folder.

The core of the benchmark is represented by `bash run_cpp.sh`, `bash run_py.sh``bash run_omp.sh`. These scripts execute the respective models, storing their output in the *out/* folder and their time measures in the *measures/* folder. You can control the number of OMP threads acting on *params.sh*.

Finally, execute `bash elaborate.sh`, it will compare each C++ output with the correspondent output of the other two models ensuring that all of them behave the same. Each comparison will be stored in the *out_comparison/* folder, but they will also be collapsed in a single file in the *logs/* folder. This same script will analyze time measures as well, and the result it gets will again be put in the *logs/* folder.

At the end of the benchmark, you will have everything you need in the *logs/* folder:

- *dataset_info.txt*: Contains the parameters used to create the dataset.
- *model_info.txt*: Contains the main attributes of the models used.
- *output_analysis.txt*: Here you can find the comparison pf the outputs. Use this file to understand if the models produce the same predictions or behave differently.
- *measures_analysis.txt*: This file contains some time statistics for each model. You can use it to understand which model performs better.

When you want to clean the work space, you have two possible commands:

- `make clean`: removes the bin folders, the obj folders, *out_comparison/* and *logs/*, but it leaves the dataset and the models untouched. It is used to perform different benchmarks on the same dataset-model pair.
- `make clean_everything`: removes all the generated folder, leaving the folder as just cloned. It is used when you also want a new dataset and a new model for your next benchmark.

## Maintainer

*Alex Pegoraro*

# GPU acceleration
This section is for explain briefly how are made and how to use the Accelerated components

## Repo Structure
- *gpu_include:* this folder contains all the header files following the same logic as the cpp section.
- *gpu_src:* this folder contains all the source files with the C++/CUDA implementations following the same logic as the cpp section.
- *test_src:* in the same folder as the previous section there are the test implementations, they are build with the same commands as cpp

> [!Warning]
> Actually is implemented only the Conv2d kernels used in the patch embedder.

## Conv2d
To execute the tests run
```bash
make test_bin/test_gpu_conv2d.exe
./test_bin/test_gpu_conv2d.exe <level>
```
where `level` is a value between 0-2 that indicates which kernel I want to launch. (0 single_channel, 1 unified or 2 stream parallelized).

## Norm (Layer Norm)

## Attention

## Layer Scale

## MLP
