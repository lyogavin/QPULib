#include <QPULib.h>
#include <cstring>


// ============================================================================
// Main
// ============================================================================

void conv1x1s1_sgemm_cpu(float* bottom_blob, float* top_blob, float* kernel, float* bias, int w, int h, int inch, int outch)
{
    for (int k = 0; k < outch; k++) {
        for (int j = 0; j < inch; j++) {
            for (int i = 0; i < w; i++) {
                for (int l = 0; l < h; l++) {
                    top_blob[w*h*k + w*l + i] = bias[k];
                }
            }
            for (int i = 0; i < w; i++) {
                for (int l = 0; l < h; l++) {
                    top_blob[w*h*k + w*l + i] += bottom_blob[w*h*j + w*l + i] * kernel[k * inch + j];
                }
            }
        }
    }
}

//void conv1x1s1_sgemm_qpu(void* bottom_blob, void* top_blob, void* kernel, void* bias, int w, int h, int inch, int outch, int elemsize)

float get_diff(float* input, float* output, int size) {
    float diff = 0;
    for(int i = 0;i < size;i++) {
        diff += output[i] - input[i];
    }

    return diff;
}

int main()
{
  // Timestamps
  timeval tvStart, tvEnd, tvDiff;

  // Number of vertices and angle of rotation
  const int N = 19200; // 192000
  const float THETA = (float) 3.14159;

  const int w = 10, h=10, inch=10, outch=10,elemsize=4;

  float* bot = new float(w*h*inch);
  float* top = new float(w*h*outch);
  float* ker = new float(outch*inch);
  float* bias = new float(outch);


  float* botcpu = new float(w*h*inch);
  memcpy(botcpu, bot, w*h*inch);
  float* topcpu = new float(w*h*outch);
  memcpy(topcpu, top, w*h*outch);
  float* kercpu = new float(outch*inch);
  memcpy(kercpu, ker, outch*inch);
  float* biascpu = new float(outch);
  memcpy(biascpu, bias, outch);

  conv1x1s1_sgemm_cpu(bot, top, ker, bias, w, h, inch, outch);



  gettimeofday(&tvStart, NULL);

  conv1x1s1_sgemm_qpu(botcpu, topcpu, kercpu, biascpu, w, h, inch, outch, sizeof(float));

  gettimeofday(&tvEnd, NULL);
  timersub(&tvEnd, &tvStart, &tvDiff);

  // Display results
  //for (int i = 0; i < N; i++)
  //  printf("%f %f\n", x[i], y[i]);

  float diff = get_diff(top, topcpu, w*h*outch);
 
  printf("diff: %f, %ld.%06lds\n", diff, tvDiff.tv_sec, tvDiff.tv_usec);

  return 0;
}
