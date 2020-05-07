#include "App/Sgemm.h"

#include <cstring>

void conv1x1s1_sgemm_qpulib(Ptr<Float> bottom, Ptr<Float> top, Ptr<Float> kernel, Ptr<Float> bias,
                                   Int w, Int h, Int inch, Int outch, Int elemsize)
{
    // 1. multiple QPU...
    Int outch_inc = numQPUs();

    Print("numQPUs:");
    Print(outch_inc);
    Print("\n");

    Int inc = 16;//numQPUs() << 4;

    Float kernel_last;
    Float bias_last;

    For (Int k = me(), k < outch, k = k + outch_inc)
        Ptr<Float> kernel_ptr = kernel + k * inch;
        Ptr<Float> bias_ptr = bias + k;
        gather(bias_ptr);
        receive(bias_last);
        //bias_last = *bias_ptr;

        For (Int j = 0, j < inch, j = j + 1)
            Ptr<Float> top_ptr = top + index() + k * w * h;

            gather(kernel_ptr);
            receive(kernel_last);
            //kernel_last = *kernel_ptr;

            //Int i  = 0;

            Ptr<Float> bottom_ptr = bottom + index() + w * h * j;

            Float bottom_last;
            Float top_last;

            gather(bottom_ptr);
            gather(top_ptr);



            Int last_i = -1;



            For (Int i = 0, i + inc - 1 < (w * h), i = i + inc)
                last_i = i + inc - 1;
                //If (i + inc + inc - 1 < w * h)
                gather(bottom_ptr + inc);
                gather(top_ptr + inc);
                //End
                receive(bottom_last);
                receive(top_last);

                If (j == 0)
                    store(bottom_last * kernel_last + bias_last, top_ptr);
                Else
                    store(bottom_last * kernel_last + top_last, top_ptr);
                End
                bottom_ptr = bottom_ptr + inc;
                top_ptr = top_ptr + inc;
            End

            // gather the rest one by one
            receive(bottom_last);
            receive(top_last);

            top_last = *top_ptr;

            /*
            Print("j:");
            Print(j);
            Print("\n");

            Print("last_i+1:");
            Print(last_i+1);
            Print("\n");


            Print("top_last");
            Print(toInt(top_last));
            Print("\n");*/

            If (last_i + 1 < w * h)
                Int left_len = w * h - last_i - 1;
                Float all_one = 1.0f;
                Float all_zero = 0;

                Where(index() < left_len)
                    all_zero = all_one;
                End


                If (j == 0)
                    /*
                    Int to_store = toInt(all_zero * (bottom_last * kernel_last + bias_last) + (all_one - all_zero) * top_last);

                    Print("to_store");
                    Print(to_store);
                    Print("\n");

                    Print("all zero:");
                    Print(toInt(all_zero));
                    Print(",");

                    Print("all_one");
                    Print(toInt(all_one));
                    Print("\n");
                    */

                    store(all_zero * (bottom_last * kernel_last + bias_last) + (all_one - all_zero) * top_last, top_ptr);
                    //*top_ptr = all_zero * (bottom_last * kernel_last + bias_last) + (all_one - all_zero) * top_last;
                    flush();
                    //store(all_zero * (bottom_last * kernel_last + bias_last) + (all_one - all_zero) * top_last, top_ptr);

                    /*
                    Print("stored");
                    gather(top_ptr);
                    Float temp;
                    receive(temp);
                    Print(toInt(temp));
                    Print("\n");
                    */
                Else
                    /*
                    Int to_store = toInt(all_zero * (bottom_last * kernel_last + top_last) + (all_one - all_zero) * top_last);

                    Print("to_store");
                    Print(to_store);
                    Print("\n");

                    Print("all zero:");
                    Print(toInt(all_zero));
                    Print(",");

                    Print("all_one");
                    Print(toInt(all_one));
                    Print("\n");
                    */

                    store(all_zero * (bottom_last * kernel_last + top_last) + (all_one - all_zero) * top_last, top_ptr);
                    //*top_ptr = all_zero * (bottom_last * kernel_last + top_last) + (all_one - all_zero) * top_last;
                    flush();
                    //store(all_zero * (bottom_last * kernel_last + top_last) + (all_one - all_zero) * top_last, top_ptr);

                    /*
                    Print("stored");
                    gather(top_ptr);
                    Float temp;
                    receive(temp);
                    Print(toInt(temp));
                    Print("\n");
                    */

                End
            End

            kernel_ptr = kernel_ptr + 1;
            //bias_ptr = bias_ptr + 1;
        End
    End

    // Discard pre-fetched vectors from final iteration
    //receive(kernel_last);
    //receive(bottom_last);
    //receive(top_last);
}


void memcpy_to_shared(SharedArray<float>* dest, float* src, unsigned size)
{
    for (int i =0; i<size; i++){
        (*dest)[i] = src[i];
    }
}


void memcpy_from_shared(float* dest, SharedArray<float>* src, unsigned size)
{
    for (int i =0; i<size; i++){
        dest[i] = (*src)[i];
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
    memcpy_to_shared(&bottom_shar, bottom_blob, total * inch);
    printf("alloc top");
    SharedArray<float> top_shar(total * outch + padding);

    for (int i =0; i<total * outch + padding; i++){
        top_shar[i] = 0;
    }

    //memcpy_to_shared(&top_shar, top_blob, total * outch);
    printf("alloc kernel");
    SharedArray<float> kernel_shar(inch * outch + padding);
    memcpy_to_shared(&kernel_shar, kernel, inch * outch);
    printf("alloc bias");
    SharedArray<float> bias_shar(outch + padding);
    memcpy_to_shared(&bias_shar, bias, outch);

    // Compile kernel
    auto k = compile(conv1x1s1_sgemm_qpulib);

    // Invoke kernel
    k.setNumQPUs(NQPUS);

    k(&bottom_shar, &top_shar, &kernel_shar, &bias_shar, w, h, inch, outch, elemsize);

    memcpy_from_shared(top_blob, &top_shar, total*outch);

}