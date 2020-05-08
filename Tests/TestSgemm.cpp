#include <QPULib.h>
#include <cstring>

#include <cstdlib>

#include <ctime>
#include <math.h>
using namespace std;


// ============================================================================
// Main
// ============================================================================

void conv1x1s1_sgemm_cpu(float* bottom_blob, float* top_blob, float* kernel, float* bias, int w, int h, int inch, int outch)
{
    for (int k = 0; k < outch; k++) {
        for (int j = 0; j < inch; j++) {
            /*for (int i = 0; i < w; i++) {
                for (int l = 0; l < h; l++) {
                    top_blob[w*h*k + w*l + i] = bias[k];
                }
            }*/
            for (int i = 0; i < w; i++) {
                for (int l = 0; l < h; l++) {
                    if (j == 0) {
                        top_blob[w*h*k + w*l + i] = bottom_blob[w*h*j + w*l + i] * kernel[k * inch + j] + bias[k];
                    } else {
                        top_blob[w*h*k + w*l + i] += bottom_blob[w*h*j + w*l + i] * kernel[k * inch + j];
                    }
                }
            }
        }
    }
}

//void conv1x1s1_sgemm_qpu(void* bottom_blob, void* top_blob, void* kernel, void* bias, int w, int h, int inch, int outch, int elemsize)

float get_diff(float* input, float* output, int size) {
    float diff = 0;
    for(int i = 0;i < size;i++) {
        float single_diff = output[i] - input[i];
        if (fabsf(single_diff) > 0.0001f){
            printf("diff pos: %d, %f - %f", i, input[i], output[i]);
        }
        diff += single_diff;
    }

    return diff;
}
void print_array(float* array, int size)
{
    for (int i =0; i< size; i++) {
        printf("%f\t", array[i]);
    }
    printf("\n");
}

void fill_rand(float* dest, int size)
{
    for(int i =0; i<size;i++)
    {
        dest[i] = rand()/float(RAND_MAX);
    }
    print_array(dest, size);
}


int main()
{
  // Timestamps
  timeval tvStart, tvEnd, tvDiff;
  timeval tvStartQpu, tvEndQpu, tvDiffQpu;

  // Number of vertices and angle of rotation
  const float THETA = (float) 3.14159;

  //const int w = 2, h=2, inch=2, outch=2,elemsize=4;
  //const int w = 150, h=150, inch=16, outch=96,elemsize=4;
  const int w = 150, h=150, inch=1, outch=1,elemsize=4;

  float* bot = new float[w*h*inch];
  printf("bot:\n");
  fill_rand(bot, w*h*inch);
  float* top = new float[w*h*outch];
  printf("top:\n");
  fill_rand(top, w*h*outch);
  float* ker = new float[outch*inch];
  printf("ker:\n");
  fill_rand(ker, outch*inch);
  float* bias = new float[outch];
  printf("bias:\n");
  fill_rand(bias, outch);

  float* topcpu = new float[w*h*outch];
  memcpy(topcpu, top, w*h*outch*sizeof(float));

  float* debug_output = new float[1000];

  srand(time(0));



  gettimeofday(&tvStart, NULL);

  conv1x1s1_sgemm_cpu(bot, topcpu, ker, bias, w, h, inch, outch);

  gettimeofday(&tvEnd, NULL);
  timersub(&tvEnd, &tvStart, &tvDiff);

  printf("cpu: %ld.%06lds\n", tvDiff.tv_sec, tvDiff.tv_usec);


  //topcpu[3] = 10.0f;


  gettimeofday(&tvStartQpu, NULL);

  printf("starting conv1x1s1_sgemm_qpu");
  conv1x1s1_sgemm_qpu(bot, top, ker, bias, debug_output, 1000, w, h, inch, outch, sizeof(float));

  gettimeofday(&tvEndQpu, NULL);
  timersub(&tvEndQpu, &tvStartQpu, &tvDiffQpu);

  // Display results
  const int N = 20; // 192000
  for (int i = 0; i < N; i++)
      printf("top:%f topcpu:%f\n", top[i], topcpu[i]);

  for (int i =0;i<50;i++)
      printf("%d - offset: %d, value: %f\n", i, int(debug_output[2*i]), debug_output[2*i+1]);


  float diff = get_diff(top, topcpu, w*h*outch);
 
  printf("QPU: diff: %f, %ld.%06lds\n", diff, tvDiffQpu.tv_sec, tvDiffQpu.tv_usec);

  delete bot;
  delete top;
  delete ker;
  delete bias;
  delete topcpu;

  return 0;
}
