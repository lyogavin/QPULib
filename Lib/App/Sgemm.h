#include <QPULib.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>



void conv1x1s1_sgemm_qpu(void* bottom_blob, void* top_blob, void* kernel, void* bias, int w, int h, int inch, int outch, int elemsize);