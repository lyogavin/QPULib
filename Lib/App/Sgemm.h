#include <QPULib.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include <QPULib_external.h>

//Kernel<Ptr<Float>, Ptr<Float>, Ptr<Float>, Ptr<Float>, Ptr<Float>, Int, Int, Int, Int, Int, Int>

/*void conv1x1s1_sgemm_qpu(float* bottom_blob, float* top_blob, float* kernel, float* bias,
    float* debug_output, int debug_output_size,
    int w, int h, int inch, int outch, int elemsize);*/