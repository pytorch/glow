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

#include "glow/LLVMIRCodeGen/JITFilePrinter.h"
#include "glow/LLVMIRCodeGen/LLVMIRGen.h"

#include "llvm/Support/DynamicLibrary.h"

using namespace glow;

#if defined _MSC_VER
#define EXTERN __declspec(dllexport)
#else
#define EXTERN
#endif

static JITFileWriter jitfileWriter;

int JITFileWriter::JITfprintf(FILE *f, const char *format, ...) {
  va_list argptr;
  va_start(argptr, format);
  return vfprintf(f, format, argptr);
}

FILE *JITFileWriter::JITfopen(const char *path, const char *mode) {
  // File access is locked using per file name lock. This lock is a simple
  // bool variable with it's access guarded by mutex.
  std::unique_lock<std::mutex> lck(jitfileWriter.mtx_);
  jitfileWriter.readyCV.wait(
      lck, [&path] { return !jitfileWriter.fileLock_[std::string(path)]; });
  jitfileWriter.fileLock_[std::string(path)] = true;
  FILE *f = fopen(path, mode);
  jitfileWriter.fileMap_[f] = std::string(path);
  return f;
}

int JITFileWriter::JITfclose(FILE *f) {
  std::lock_guard<std::mutex> lck(jitfileWriter.mtx_);
  if (!jitfileWriter.fileMap_.count(f)) {
    return -1;
  }
  std::string &file = jitfileWriter.fileMap_[f];
  int ret = fclose(f);
  jitfileWriter.fileMap_.erase(f);
  jitfileWriter.fileLock_[file] = false;
  jitfileWriter.readyCV.notify_all();
  return ret;
}

/// Expose host side file printer for JIT code to use.
/// The function simply exposes the writer object as a global symbol
/// that will be avaiable for JIT to use.
void LLVMIRGen::generateJITFileWriter() {
  llvm::sys::DynamicLibrary::AddSymbol("_jitFileWriter", &jitfileWriter);
}
