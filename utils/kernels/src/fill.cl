#include "utils.h"

kernel void ffill(float pattern, global float* out) {
  out[get_global_id_linear()] = pattern;
}

kernel void fadd(global const float* a, float scaleA, global const float* b,
                 float scaleB, global float* out) {
  int id = get_global_id_linear();
  out[id] = a[id] * scaleA + b[id] * scaleB;
  printf("%f + %f = %f", a[id], b[id], out[id]);
}
