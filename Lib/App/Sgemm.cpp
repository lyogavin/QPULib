#include "App/Sgemm.h"

#include <cstring>
#include <ctime>



#define output(f)  *(debug_output) = f; debug_output = debug_output + 16;


//#define DEBUG

void conv1x1s1_sgemm_qpulib(Ptr<Float> bottom, Ptr<Float> top, Ptr<Float> kernel, Ptr<Float> bias,
                            Ptr<Float> debug_output_buffer, Int debug_output_size,
                                   Int w, Int h, Int inch, Int outch, Int elemsize)
{
    //Ptr<Float> debug_output = debug_output_buffer + index();
    // 1. multiple QPU...
    Int outch_inc = numQPUs();


    Print("numQPUs:");
    Print(outch_inc);
    Print("\n");

    Int inc = 16;//numQPUs() << 4;

    Float kernel_last;
    Float bias_last;
    Float bottom_last;

    Ptr<Float> kernel_ptr = kernel;// + (k * inch);

    Ptr<Float> bias_ptr = bias;// + k;






    Ptr<Float> top_ptr;

    Ptr<Float> bottom_ptr;

    For (Int k = me(), k < outch, k = k + outch_inc)
        If (k == 1)
            Print("k*inch");
            Print(k*inch);
            Print("\n");
        End

        bias_ptr = bias + k;
        gather(bias_ptr);
        receive(bias_last);

        If (k == 1)
            Print("received bias:");
            Print(toInt(bias_last * 10000.0f));
            Print("\n");
        End

        Int offset = k* w* h;
        top_ptr = top + index() + offset;

        //For (Int i = 0, i + inc - 1 < (w * h), i = i + inc)
        For (Int i = 0, i < (w * h), i = i + inc)
            Float sum = bias_last;
            If (k == 1)
                Print("sum:");
                Print(toInt(sum * 10000.0f));
                Print("\n");
            End

            bottom_ptr = bottom + index() + i;

            Ptr<Float> kernel_ptr = kernel + (k * inch);

            gather(kernel_ptr);
            gather(bottom_ptr);



            For (Int j = 0, j < inch, j = j + 1)

                gather(kernel_ptr + 1);
                gather(bottom_ptr + w*h);

                receive(kernel_last);
                receive(bottom_last);


                If (k == 1)
                    Print("kernel using:");
                    Print(toInt(kernel_last * 10000));
                    Print("\n");

                    Print("bottom_last using:");
                    Print(toInt(bottom_last * 10000));
                    Print("\n");
                End

                sum = sum + kernel_last * bottom_last;


                If (k == 1)
                    Print("sum:");
                    Print(toInt(sum * 10000));
                    Print("\n");
                End

                kernel_ptr = kernel_ptr + 1;
                bottom_ptr = bottom_ptr + w*h;
            End
            receive(kernel_last);
            receive(bottom_last);

            If (k == 1)
                Print("sum to store:");
                Print(toInt(sum * 10000));
                Print("\n");
            End

            If (i + inc - 1 >= (w * h))
                Int exceeding_len = i + inc - (w * h);
                Float old_top = *top_ptr;

                Float to_store = 0;
                Where(index() < 16 - exceeding_len)
                    to_store = sum;
                End
                Where(index() >= 16 - exceeding_len)
                    to_store = old_top;
                End

                store(to_store, top_ptr);
            Else
                store(sum, top_ptr);
            End

            top_ptr = top_ptr + inc;
        End

        {
        /*
        For (Int j = 0, j < inch, j = j + 1)
            Int offset = k* w* h;
            top_ptr = top + index() + offset;

            gather(kernel_ptr);
            receive(kernel_last);

            bottom_ptr = bottom + index() + (w * h * j);

            Float bottom_last;
            Float top_last;

            gather(bottom_ptr);
            Print("gather:");
            Print("bottom_ptr");
            Print("\n");
            gather(top_ptr);
            Print("gather:");
            Print("top_ptr");
            Print("\n");


            Int last_i = -1;



            For (Int i = 0, i + inc - 1 < (w * h), i = i + inc)
                last_i = i + inc - 1;
                //If (i + inc + inc - 1 < w * h)
                gather(bottom_ptr + inc);
                Print("gather inside :");
                Print("bottom_ptr + inc");
                Print("\n");
                gather(top_ptr + inc);
                Print("gather inside :");
                Print("top_ptr + inc");
                Print("\n");
                //End
                receive(bottom_last);
                Print("receive inside :");
                Print("bottom_last + inc");
                Print("\n");
                output(bottom_last);

                receive(top_last);
                Print("receive inside :");
                Print("top_last + inc");
                Print("\n");
                output(top_last);

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
            //bottom_last = *bottom_ptr;
            Print("receive:");
            Print(toInt(bottom_last));
            Print("\n");

            output(bottom_last);

            // read after write doesn't work
            receive(top_last);
            top_last = *top_ptr;

            If (last_i + 1 < w * h)
                Int left_len = w * h - last_i - 1;
                Float all_one = 1.0f;
                Float all_zero = 0;

                Where(index() < left_len)
                    all_zero = all_one;
                End


                If (j == 0)

                    Int to_store = toInt(all_zero * (bottom_last * kernel_last + bias_last) + (all_one - all_zero) * top_last);

                    //store(all_zero * (bottom_last * kernel_last + bias_last) + (all_one - all_zero) * top_last, top_ptr);
                    *top_ptr = all_zero * (bottom_last * kernel_last + bias_last) + (all_one - all_zero) * top_last;



                Else

                    Int to_store = toInt(all_zero * (bottom_last * kernel_last + top_last) + (all_one - all_zero) * top_last);

                    //store(all_zero * (bottom_last * kernel_last + top_last) + (all_one - all_zero) * top_last, top_ptr);
                    *top_ptr = all_zero * (bottom_last * kernel_last + top_last) + (all_one - all_zero) * top_last;
                End
            End

            kernel_ptr = kernel_ptr + 1;
            //bias_ptr = bias_ptr + 1;
        End

        //kernel_ptr = kernel_ptr + inch;
        bias_ptr = bias_ptr + 1;
        */
        }
    End

    flush();

    // Discard pre-fetched vectors from final iteration
    //receive(kernel_last);
    //receive(bottom_last);
    //receive(top_last);
}


/*auto get_qpu_sgemm_kernel()
{
    return compile(conv1x1s1_sgemm_qpulib);
}*/

static SgemmKernel* compiled_sgemm_kernel = NULL;

void init_qpulib_sgemm()
{
    static SgemmKernel instance = compile(conv1x1s1_sgemm_qpulib);
    compiled_sgemm_kernel = &instance;
}



void memcpy_to_shared(SharedArray<float>* dest, float* src, unsigned size)
{
    for (int i =0; i<size; i++){
        (*dest)[i] = src[i];
#ifdef DEBUG
        printf("%f\t", src[i]);
#endif
    }
#ifdef DEBUG
    printf("\n");
#endif

}


void memcpy_to_shared_expand(SharedArray<float>* dest, float* src, unsigned size)
{
    for (int i =0; i<size; i++){
#ifdef DEBUG
        printf("%f\t", src[i]);
#endif
        for (int j=i*16;j<i*16+16;j++){
            (*dest)[j] = src[i];
        }
    }
#ifdef DEBUG
    printf("\n");
#endif
}

void memcpy_from_shared(float* dest, SharedArray<float>* src, unsigned size)
{
    for (int i =0; i<size; i++){
        dest[i] = (*src)[i];
    }
}

void conv1x1s1_sgemm_qpu(float* bottom_blob, float* top_blob, float* kernel, float* bias,
    float* debug_output, int debug_output_size,
    int w, int h, int inch, int outch, int elemsize)
{
    int padding = 16;
    int total = w * h;
    int NQPUS = 12;

    // Timestamps
    timeval tvStart, tvEnd, tvDiff;

    gettimeofday(&tvStart, NULL);

    // 1. copy data to shared memeory...
#ifdef DEBUG
    printf("alloc bottom");
#endif
    SharedArray<float> bottom_shar(total * inch + padding);
    memcpy_to_shared(&bottom_shar, bottom_blob, total * inch);
#ifdef DEBUG
    printf("alloc top");
#endif
    SharedArray<float> top_shar(total * outch + padding);

    for (int i =0; i<total * outch + padding; i++){
        top_shar[i] = 0;
    }

    //memcpy_to_shared(&top_shar, top_blob, total * outch);
#ifdef DEBUG
    printf("alloc kernel");
#endif
    SharedArray<float> kernel_shar(inch * outch + padding);
    memcpy_to_shared(&kernel_shar, kernel, inch * outch);
#ifdef DEBUG
    printf("alloc bias");
#endif
    SharedArray<float> bias_shar(outch + padding);
    memcpy_to_shared(&bias_shar, bias, outch);

    if (debug_output_size > 0)
        SharedArray<float> debug_output_shar(debug_output_size);

    gettimeofday(&tvEnd, NULL);
    timersub(&tvEnd, &tvStart, &tvDiff);
    printf("memory operation time: %ld.%06lds\n", tvDiff.tv_sec, tvDiff.tv_usec);

    // Compile kernel
    auto k = *compiled_sgemm_kernel;

    // Invoke kernel
    k.setNumQPUs(NQPUS);

    k(&bottom_shar, &top_shar, &kernel_shar, &bias_shar, &debug_output_shar, debug_output_size, w, h, inch, outch, elemsize);

    memcpy_from_shared(top_blob, &top_shar, total*outch);

    if (debug_output_size > 0)
        memcpy_from_shared(debug_output, &debug_output_shar, debug_output_size);

}