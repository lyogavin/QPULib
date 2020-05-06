#include "App/Sgemm.h"

#include <string.h>


void conv1x1s1_sgemm_qpulib(Ptr<Float> bottom, Ptr<Float> top, Ptr<Float> kernel, Ptr<Float> bias,
                                   int w, int h, int inch, int outch, int elemsize)
{
    // 1. multiple QPU...
    Int outch_inc = numQPUs();

    Int inc = 16;//numQPUs() << 4;

    Float bottom_last;
    Float kernel_last;
    Float top_last;
    Float bias_last;

    For (Int k = me(), k < outch, k = k + outch_inc)
        Ptr<Float> kernel_ptr = kernel + k * inch;
        Ptr<Float> bias_ptr = bias + k;

        For (Int j = 0, j < inch, j = j + 1)
            Ptr<Float> top_ptr = top + index() + k * w * h;

            gather(kernel_ptr);
            receive(kernel_last);
            gather(bias_ptr);
            receive(bias_last);

            Int i  = 0;

            Ptr<Float> bottom_ptr = bottom + index() + w * h * j;

            gather(bottom_ptr);
            gather(top_ptr);

            For (i = 0, i < w * h, i = i + inc)
                gather(bottom_ptr + inc);
                gather(top_ptr + inc);
                receive(bottom_last);
                receive(top_last);

                store(bottom_last * kernel_last + top_last, top_ptr);

                bottom_ptr = bottom_ptr + inc;
                top_ptr = top_ptr + inc;
            End

            // gather the rest one by one
            receive(bottom_last);
            receive(top_last);

            i =  i - inc;

            Ptr<Float> bottom_ptr_by_one = bottom + i + w * h * j;
            gather(bottom_ptr_by_one);
            Ptr<Float> top_ptr_by_one = top + i + k * w * h;
            gather(top_ptr_by_one);

            Float bottom_last_by_one;
            Float top_last_by_one;

            For (, i < w * h, i = i + 1)
                gather(bottom_ptr_by_one + 1);
                gather(top_ptr_by_one + 1);
                receive(bottom_last_by_one);
                receive(top_last_by_one);

                store(bottom_last_by_one * kernel_last + top_last_by_one, top_ptr_by_one);

                bottom_ptr_by_one = bottom_ptr_by_one + 1;
                top_ptr_by_one = top_ptr_by_one + 1;
            End
            receive(bottom_last_by_one);
            receive(top_last_by_one);

            kernel_ptr = kernel_ptr + 1;
            bias_ptr = bias_ptr + 1;
        End
    End

    // Discard pre-fetched vectors from final iteration
    //receive(kernel_last);
    //receive(bottom_last);
    //receive(top_last);
}

void conv1x1s1_sgemm_qpu(void* bottom_blob, void* top_blob, void* kernel, void* bias, int w, int h, int inch, int outch, int elemsize)
{
    int padding = 16;
    int total = w * h * elemsize;
    int NQPUS = 12;
    // 1. copy data to shared memeory...
    SharedArray<float> bottom_shar(total * inch + padding);
    memcpy(bottom_shar.getPointer(), bottom_blob, total * inch);
    SharedArray<float> top_shar(total * outch + padding);
    memcpy(top_shar.getPointer(), top_blob, total * outch)
    SharedArray<float> kernel_shar(inch * outch * elemsize + padding);
    memcpy(kernel_shar.getPointer(), kernel, inch * outch * elemsize);
    SharedArray<float> bias_shar(outch * elemsize + padding);
    memcpy(bias_shar.getPointer(), bias, outch * elemsize);

    // Compile kernel
    auto k = compile(conv1x1s1_sgemm_qpulib);

    // Invoke kernel
    k.setNumQPUs(NQPUS);

    k(&bottom_shar, &top_shar, &kernel_shar, &bias_shar, w, h, inch, outch, elemsize);

}