#include "../gpu_include/cudnn_utils.h"

static const char* axisToStr(cudnnSeqDataAxis_t a) {
  switch (a) {
    case CUDNN_SEQDATA_TIME_DIM:  return "TIME";
    case CUDNN_SEQDATA_BATCH_DIM: return "BATCH";
    case CUDNN_SEQDATA_BEAM_DIM:  return "BEAM";
    case CUDNN_SEQDATA_VECT_DIM:  return "VECT";
    default: return "UNKNOWN";
  }
}

static size_t dataTypeSize(cudnnDataType_t t) {
  switch (t) {
    case CUDNN_DATA_HALF:   return 2;
    case CUDNN_DATA_FLOAT:  return 4;
    case CUDNN_DATA_DOUBLE: return 8;
    case CUDNN_DATA_INT8:   return 1;
    case CUDNN_DATA_INT32:  return 4;
    case CUDNN_DATA_INT64:  return 8;
    // add others you use
    default: return 0;
  }
}

void set_attr(cudnnBackendDescriptor_t d, cudnnBackendAttributeName_t name,
                     cudnnBackendAttributeType_t type, int64_t n, const void* ptr) {
  CUDNN_CHECK(cudnnBackendSetAttribute(d, name, type, n, ptr));
}
void finalize(cudnnBackendDescriptor_t d) {
  CUDNN_CHECK(cudnnBackendFinalize(d));
}

void printSeqDataDescriptor(cudnnSeqDataDescriptor_t desc)
{
  cudnnDataType_t dataType = CUDNN_DATA_FLOAT;
  int nbDims = 0;

  // cuDNN header defines this as 4 in v8 (TIME,BATCH,BEAM,VECT). :contentReference[oaicite:1]{index=1}
  int dimA[CUDNN_SEQDATA_DIM_COUNT];
  cudnnSeqDataAxis_t axes[CUDNN_SEQDATA_DIM_COUNT];

  size_t seqLengthArraySize = 0;

  // --- Pass 1: query how many sequence-length entries exist (no buffer yet)
  CUDNN_CHECK(cudnnGetSeqDataDescriptor(
      desc,
      &dataType,
      &nbDims,
      CUDNN_SEQDATA_DIM_COUNT,   // nbDimsRequested
      dimA,
      axes,
      &seqLengthArraySize,
      0,                         // seqLengthSizeRequested (ignored because seqLengthArray=NULL)
      NULL,                      // seqLengthArray
      NULL                       // paddingFill (optional)
  ));

  int *seqLengths = NULL;
  if (seqLengthArraySize > 0) {
    seqLengths = (int*)malloc(seqLengthArraySize * sizeof(int));
    if (!seqLengths) {
      fprintf(stderr, "malloc failed\n");
      exit(1);
    }
  }

  // paddingFill: must be big enough for the datatype stored in the descriptor
  unsigned char paddingBuf[16];
  memset(paddingBuf, 0, sizeof(paddingBuf));

  // --- Pass 2: fetch seq lengths + paddingFill (and dims/axes again)
  CUDNN_CHECK(cudnnGetSeqDataDescriptor(
      desc,
      &dataType,
      &nbDims,
      CUDNN_SEQDATA_DIM_COUNT,
      dimA,
      axes,
      &seqLengthArraySize,
      seqLengthArraySize,        // now we provide enough space
      seqLengths,
      paddingBuf
  )); // prototype matches the one you posted :contentReference[oaicite:2]{index=2}

  printf("cudnnSeqDataDescriptor_t\n");
  printf("  dataType=%d\n", (int)dataType);
  printf("  nbDims=%d\n", nbDims);
  for (int i = 0; i < nbDims; ++i) {
    printf("  dim[%d]: axis=%s (%d) size=%d\n", i, axisToStr(axes[i]), (int)axes[i], dimA[i]);
  }

  printf("  seqLengthArraySize=%zu\n", seqLengthArraySize);
  if (seqLengths && seqLengthArraySize > 0) {
    for (size_t i = 0; i < seqLengthArraySize; ++i) {
      printf("    seqLen[%zu]=%d\n", i, seqLengths[i]);
    }
  }

  // Print paddingFill in a type-aware way (common cases)
  size_t tsz = dataTypeSize(dataType);
  printf("  paddingFill (raw %zu bytes):", tsz);
  for (size_t i = 0; i < tsz && i < sizeof(paddingBuf); ++i) printf(" %02X", paddingBuf[i]);
  printf("\n");

  if (dataType == CUDNN_DATA_FLOAT) {
    float v; memcpy(&v, paddingBuf, sizeof(v));
    printf("  paddingFill (float)=%f\n", v);
  } else if (dataType == CUDNN_DATA_DOUBLE) {
    double v; memcpy(&v, paddingBuf, sizeof(v));
    printf("  paddingFill (double)=%lf\n", v);
  }

  free(seqLengths);
}
