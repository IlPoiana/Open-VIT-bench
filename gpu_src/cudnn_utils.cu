#include "../gpu_include/cudnn_utils.h"

void set_attr(cudnnBackendDescriptor_t d, cudnnBackendAttributeName_t name,
                     cudnnBackendAttributeType_t type, int64_t n, const void* ptr) {
  CUDNN_CHECK(cudnnBackendSetAttribute(d, name, type, n, ptr));
}
void finalize(cudnnBackendDescriptor_t d) {
  CUDNN_CHECK(cudnnBackendFinalize(d));
}