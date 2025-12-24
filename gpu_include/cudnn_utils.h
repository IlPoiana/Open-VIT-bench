#pragma once
#include <iostream>
#include <cuda_runtime.h>
#include <cudnn_backend.h>
#include <cudnn.h>
#include "../gpu_include/gpu_datatypes.h"
 
void set_attr(cudnnBackendDescriptor_t d, cudnnBackendAttributeName_t name,
                    cudnnBackendAttributeType_t type, int64_t n, const void* ptr);
    
void finalize(cudnnBackendDescriptor_t d);

void printSeqDataDescriptor(cudnnSeqDataDescriptor_t desc);
