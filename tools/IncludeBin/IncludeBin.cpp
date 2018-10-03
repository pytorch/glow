#include <cstdio>
#include <cstdlib>

int main(int argc, char **argv) {
  if (argc != 3) {
    fprintf(stderr, "Usage: bin-include input_file output_file\n");
    exit(1);
  }
  FILE *in = fopen(argv[1], "rb");
  if (!in) {
    perror("Could not open input file");
    exit(1);
  }
  FILE *out = fopen(argv[2], "wb");
  if (!out) {
    perror("Could not open output file");
    exit(1);
  }
  for (int i = 0;; i++) {
    int ch = fgetc(in);
    if (ch == EOF) {
      break;
    }
    fprintf(out, " 0x%02x,", ch);
    if (i % 12 == 11) {
      fprintf(out, "\n");
    }
  }
  fprintf(out, "\n");
  fclose(out);
  fclose(in);
  return 0;
}
