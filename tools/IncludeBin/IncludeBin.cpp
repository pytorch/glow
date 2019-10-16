/**
 * Copyright (c) Glow Contributors. See CONTRIBUTORS file.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cstdio>
#include <cstdlib>

/// A simple reimplementation of `xxd -i`.
///
/// Usage: include-bin input_file output_file.
///
/// Given a binary input file, write an array of hex characters suitable for
/// inclusion as a C unsigned-char array.  The output looks like:
///
/// 0x2f, 0xc5, 0x35, 0x0c, 0xef, 0x7d, 0xea, 0x20, 0x9a, 0x80, 0x31, 0xfa,
/// 0xdd, 0x98, 0x5e, 0x95, 0xcc, 0xa3, 0xfe, 0x5a, 0xa2, 0x8f, 0xcb, 0x16,
/// 0x55, 0x14, 0x25, 0xaa, 0xfe, 0x61, 0x0e, 0xb7,
int main(int argc, char **argv) {
  if (argc != 3) {
    fprintf(stderr, "Usage: include-bin input_file output_file\n");
    exit(1);
  }
  FILE *in = fopen(argv[1], "rb");
  if (!in) {
    perror("Could not open input file");
    exit(1);
  }
  FILE *out = fopen(argv[2], "w");
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
