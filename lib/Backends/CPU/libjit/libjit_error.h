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
#ifndef GLOW_BACKENDS_CPU_LIBJIT_LIBJIT_ERROR_H
#define GLOW_BACKENDS_CPU_LIBJIT_LIBJIT_ERROR_H

/// \file libjit_error.h
/// This file contains the error codes used for the LIBJIT error handling.
/// NOTE: Make sure these error codes are synchronized with the ones defined
/// in "glow\lib\LLVMIRCodeGen\BundleSaver.cpp" which are used for printing in
/// the automatically generated header file when compiling a bundle.

#define GLOW_BUNDLE_SUCCESS 0

#endif // GLOW_BACKENDS_CPU_LIBJIT_LIBJIT_ERROR_H
