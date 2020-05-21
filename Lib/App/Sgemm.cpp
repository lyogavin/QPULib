#include "App/Sgemm.h"

#include <cstring>
#include <ctime>


//#define OLD_OVERWRITE
#define output(f)  *(debug_output) = f; debug_output = debug_output + 16;


//#define DEBUG
#define DEBUGMEM

void conv1x1s1_sgemm_qpulib(Ptr<Float> bottom, Ptr<Float> top, Ptr<Float> kernel, Ptr<Float> bias,
                            Ptr<Float> debug_output_buffer, Int debug_output_size,
                                   Int w, Int h, Int padded_total, Int inch, Int outch, Int elemsize)
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
        If (k == -1)
            Print("k*inch");
            Print(k*inch);
            Print("\n");
        End

        bias_ptr = bias + k;
        gather(bias_ptr);
        receive(bias_last);

        If (k == -1)
            Print("received bias:");
            Print(toInt(bias_last * 10000.0f));
            Print("\n");
        End

        Int offset = k* padded_total;
        Int top_ptr_offset = 0;
        Int last_top_ptr_offset = 0;
        top_ptr = top + index() + offset;
        top_ptr_offset = offset;

        //For (Int i = 0, i + inc - 1 < (w * h), i = i + inc)
        For (Int i = 0, i < (w * h), i = i + inc)
            Float sum = bias_last;
            If (k == -1)
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


                If (k == -1)
                    Print("kernel using:");
                    Print(toInt(kernel_last * 10000));
                    Print("\n");

                    Print("bottom_last using:");
                    Print(toInt(bottom_last * 10000));
                    Print("\n");
                End

                sum = sum + kernel_last * bottom_last;


                If (k == -1)
                    Print("sum:");
                    Print(toInt(sum * 10000));
                    Print("\n");
                End

                kernel_ptr = kernel_ptr + 1;
                bottom_ptr = bottom_ptr + w*h;
            End
            receive(kernel_last);
            receive(bottom_last);

            If (k == -1)
                Print("sum to store:");
                Print(toInt(sum * 10000));
                Print("\n");
            End

            If (i + inc - 1 >= (w * h))
                Int exceeding_len = i + inc - (w * h);
                // check if address trying to read overlap with last store, if so, need a flush

                //If (last_top_ptr_offset - top_ptr_offset < 16 && top_ptr_offset - last_top_ptr_offset < 16)
                //    flush();
                //End

#ifdef OLD_OVERWRITE
                Float old_top = *top_ptr;

                Float to_store = 0;
                Where(index() < 16 - exceeding_len)
                    to_store = sum;
                End
                Where(index() >= 16 - exceeding_len)
                    to_store = old_top;
                End
#else
                Float to_store = sum;
#endif
                store(to_store, top_ptr);
                last_top_ptr_offset = top_ptr_offset;
            Else
                store(sum, top_ptr);
                last_top_ptr_offset = top_ptr_offset;
            End

            top_ptr = top_ptr + inc;
            top_ptr_offset = top_ptr_offset + inc;
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

static timeval tvTotal;

void init_qpulib_sgemm()
{
    if (compiled_sgemm_kernel == NULL) {
        static SgemmKernel instance = compile(conv1x1s1_sgemm_qpulib);
        compiled_sgemm_kernel = &instance;
        timerclear(&tvTotal);
    }
}



void memcpy_to_shared(SharedArray<float>* dest, float* src, unsigned total, unsigned cstep, unsigned c)
{
    int j = 0;
    for (int i =0; i<total*c; i++){
        (*dest)[i] = src[j];
        if (j % cstep == total - 1){
            j+= cstep - total + 1;
        } else {
            j++;
        }
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
void memcpy_from_shared(float* dest, SharedArray<float>* src, unsigned padded_total, unsigned total, unsigned outcstep, unsigned outch) {

    int p_dest = 0, p_src=0;
    for (int i =0; i<outch; i++){
        p_src = i * padded_total;
        p_dest = i * outcstep;
        for (int j =0; j<total; j++){
            dest[p_dest++] = (*src)[p_src++];
        }
    }
}


void conv1x1s1_sgemm_qpu(float* bottom_blob, float* top_blob, float* kernel, float* bias,
    float* debug_output, int debug_output_size,
    int w, int h, int inch, int outch, int incstep, int outcstep, int elemsize)
{
    // preallocate shared array...
    int padding = 16;
    static int bottom_presize = 810016 + padding;// 150 * 150 * 32 + padding;
    static int top_presize = 2161168 + padding;
    static int kernel_presize = 409616 + padding;
    static int bias_presize = 1296 + padding;
    static SharedArray<float> bottom_shar(bottom_presize);
    static SharedArray<float> top_shar(top_presize);
    static SharedArray<float> kernel_shar(kernel_presize);
    static SharedArray<float> bias_shar(bias_presize);

    int total = w * h;
    int NQPUS = 12;

    // Timestamps
    timeval tvStart, tvEnd, tvDiff, tvDiff1;

    gettimeofday(&tvStart, NULL);

    // 1. copy data to shared memeory...
#ifdef DEBUG
    printf("alloc bottom");
#endif
    //SharedArray<float> bottom_shar(total * inch + padding);
    if (total * inch + padding > bottom_presize) {
        printf("bottom preallocate size %d smaller than needed, reallocate: %d\n", bottom_presize, total * inch + padding);
        bottom_presize = total * inch + padding;
        bottom_shar.dealloc();
        bottom_shar.alloc(bottom_presize);
    }
    memcpy_to_shared(&bottom_shar, bottom_blob, total, incstep, inch);
#ifdef DEBUG
    printf("alloc top");
#endif

    int padded_total = total + (total % 16 > 0 ? 16 - (total % 16) : 0);
    //SharedArray<float> top_shar(padded_total * outch + padding);
    if (padded_total * outch + padding > top_presize) {
        printf("top preallocate size %d smaller than needed, reallocate: %d\n", top_presize, padded_total * outch + padding);
        top_presize = padded_total * outch + padding;
        top_shar.dealloc();
        top_shar.alloc(top_presize);
    }

    for (int i =0; i<padded_total * outch + padding; i++){
        top_shar[i] = 0;
    }

    //memcpy_to_shared(&top_shar, top_blob, total * outch);
#ifdef DEBUG
    printf("alloc kernel");
#endif
    //SharedArray<float> kernel_shar(inch * outch + padding);
    if (inch * outch + padding > kernel_presize) {
        printf("kernel preallocate size %d smaller than needed, reallocate: %d\n", kernel_presize, inch * outch + padding);
        kernel_presize = inch * outch + padding;
        kernel_shar.dealloc();
        kernel_shar.alloc(kernel_presize);
    }

    memcpy_to_shared(&kernel_shar, kernel, inch * outch, inch * outch, 1);
#ifdef DEBUG
    printf("alloc bias");
#endif
    //SharedArray<float> bias_shar(outch + padding);
    if (outch + padding > bias_presize) {
        printf("bias preallocate size %d smaller than needed, reallocate: %d\n", bias_presize, outch + padding);
        bias_presize = outch + padding;
        bias_shar.dealloc();
        bias_shar.alloc(bias_presize);
    }
    memcpy_to_shared(&bias_shar, bias, outch, outch, 1);

    SharedArray<float> debug_output_shar(debug_output_size > 0? debug_output_size: 10 );

    gettimeofday(&tvEnd, NULL);
    timersub(&tvEnd, &tvStart, &tvDiff1);
    // Compile kernel
    SgemmKernel* k = compiled_sgemm_kernel;

    // Invoke kernel
    k->setNumQPUs(NQPUS);

    (*k)(&bottom_shar, &top_shar, &kernel_shar, &bias_shar, &debug_output_shar, debug_output_size, w, h, padded_total, inch, outch, elemsize);

#ifdef DEBUGMEM

    gettimeofday(&tvStart, NULL);
#endif
    memcpy_from_shared(top_blob, &top_shar, padded_total, total, outcstep, outch);

#ifdef DEBUGMEM
    if (debug_output_size > 0)
        memcpy_from_shared(debug_output, &debug_output_shar, debug_output_size);

    gettimeofday(&tvEnd, NULL);
    timersub(&tvEnd, &tvStart, &tvDiff);
    timeradd(&tvDiff, &tvDiff1, &tvDiff);
    timeradd(&tvDiff, &tvTotal, &tvTotal);


    printf("memory operation time: %ld.%06lds, total accumulated time: %ld.%06lds\n", tvDiff.tv_sec, tvDiff.tv_usec, tvTotal.tv_sec, tvTotal.tv_usec);
#endif

}