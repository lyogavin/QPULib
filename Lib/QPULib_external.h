#ifndef _QPULIB_EXT_H_
#define _QPULIB_EXT_H_

void conv1x1s1_sgemm_qpu(float* bottom_blob, float* top_blob, float* kernel, float* bias,
    float* debug_output, int debug_output_size,
    int w, int h, int inch, int outch, int elemsize);

#endif
