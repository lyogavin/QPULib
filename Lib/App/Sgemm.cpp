#include "App/Sgemm.h"

#include <cstring>

void conv1x1s1_sgemm_qpulib(Ptr<Float> bottom, Ptr<Float> top, Ptr<Float> kernel, Ptr<Float> bias,
                                   Int w, Int h, Int inch, Int outch, Int elemsize)
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
            gather(bias_ptr);
            receive(kernel_last);
            receive(bias_last);

            Int i  = 0;

            Ptr<Float> bottom_ptr = bottom + index() + w * h * j;

            gather(bottom_ptr);
            gather(top_ptr);

            For (i = 0, i + inc - 1 < w * h, i = i + inc)
                If (i + inc + inc - 1 < w * h)
                    gather(bottom_ptr + inc);
                    gather(top_ptr + inc);
                End
                receive(bottom_last);
                receive(top_last);

                store(bottom_last * kernel_last + top_last, top_ptr);

                bottom_ptr = bottom_ptr + inc;
                top_ptr = top_ptr + inc;
            End

            // gather the rest one by one
            //receive(bottom_last);
            //receive(top_last);

            //i =  i - inc + 1;

            Ptr<Float> bottom_ptr_by_one = bottom + i + w * h * j;
            gather(bottom_ptr_by_one);
            Ptr<Float> top_ptr_by_one = top + i + k * w * h;
            gather(top_ptr_by_one);

            Float bottom_last_by_one;
            Float top_last_by_one;

            For (, i < w * h, i = i + 1)
                If (i + 1 < w * h)
                    gather(bottom_ptr_by_one + 1);
                    gather(top_ptr_by_one + 1);
                End
                receive(bottom_last_by_one);
                receive(top_last_by_one);

                store(bottom_last_by_one * kernel_last + top_last_by_one, top_ptr_by_one);
                bottom_ptr_by_one = bottom_ptr_by_one + 1;
                top_ptr_by_one = top_ptr_by_one + 1;
            End
            //receive(bottom_last_by_one);
            //receive(top_last_by_one);

            kernel_ptr = kernel_ptr + 1;
            //bias_ptr = bias_ptr + 1;
        End
    End

    // Discard pre-fetched vectors from final iteration
    //receive(kernel_last);
    //receive(bottom_last);
    //receive(top_last);
}


void memcpy_shared(SharedArray<float>* dest, float* src, unsigned size)
{
    for (int i =0; i<size; i++){
        (*dest)[i] = src[i];
    }
}

void conv1x1s1_sgemm_qpu(void* bottom_blob, void* top_blob, void* kernel, void* bias, int w, int h, int inch, int outch, int elemsize)
{
    int padding = 16;
    int total = w * h;
    int NQPUS = 1;
    // 1. copy data to shared memeory...
    printf("alloc bottom");
    SharedArray<float> bottom_shar(total * inch + padding);
    memcpy_shared(&bottom_shar, bottom_blob, total * inch);
    printf("alloc top");
    SharedArray<float> top_shar(total * outch + padding);
    memcpy_shared(&top_shar, top_blob, total * outch);
    printf("alloc kernel");
    SharedArray<float> kernel_shar(inch * outch + padding);
    memcpy_shared(&kernel_shar, kernel, inch * outch);
    printf("alloc bias");
    SharedArray<float> bias_shar(outch + padding);
    //memcpy_shared(&bias_shar, bias, outch);

    // Compile kernel
    auto k = compile(conv1x1s1_sgemm_qpulib);

    // Invoke kernel
    k.setNumQPUs(NQPUS);

    k(&bottom_shar, &top_shar, &kernel_shar, &bias_shar, w, h, inch, outch, elemsize);

}