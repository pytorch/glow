#include <stddef.h>
#include <stdint.h>

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

void splat_f(uint8_t *buffer, size_t sz, float val) {
  for (size_t i = 0; i < sz; i++) {
    ((float *)buffer)[i] = val;
  }
}

void elementmax_f(uint8_t *dest, uint8_t *LHS, uint8_t *RHS, size_t sz) {
  for (size_t i = 0; i < sz; i++) {
    ((float *)dest)[i] = MAX(((float *)LHS)[i], ((float *)RHS)[i]);
  }
}
