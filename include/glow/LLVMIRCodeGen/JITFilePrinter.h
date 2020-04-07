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

// This file contains the print functionality the JIT code can use. The backend
// can call function to emit a global var that points to the host side struct
// containing fopen/fprintf/fclose pointers, JIT code can call instead of it's
// own fopen/fprintf/fclose.

#include <condition_variable>
#include <cstdarg>
#include <unordered_map>

/// Structure implements file I/O related functions. Currently,
/// it just maps C-lib fopen/fprintf/fclose onto it's members.
struct JITFileWriterIF {
  FILE *(*fopen)(const char *, const char *);
  int (*fprintf)(FILE *, const char *, ...);
  int (*fclose)(FILE *);
};

struct JITFileWriter {
  JITFileWriterIF IF;
  /// Open a file. Operation, arguments and return value match standard fopen
  /// function.
  static FILE *JITfopen(const char *, const char *);
  /// Writes the C string pointed by format to the stream. Operation, arguments
  /// and return value match standard fprintf function.
  static int JITfprintf(FILE *, const char *, ...);
  /// Close a file. Operation, arguments and return value match standard fclose
  /// function.
  static int JITfclose(FILE *);
  JITFileWriter() {
    IF.fopen = &JITfopen;
    IF.fprintf = &JITfprintf;
    IF.fclose = &JITfclose;
  }
  std::unordered_map<std::string, bool> fileLock_;
  std::unordered_map<FILE *, std::string> fileMap_;
  std::condition_variable readyCV;
  std::mutex mtx_;
};
