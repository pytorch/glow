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

#ifndef GLOW_TOOLS_LOADER_UTILS_H
#define GLOW_TOOLS_LOADER_UTILS_H

#include "glow/Support/Support.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <sstream>

namespace glow {

/// Typedefs for unlabeled datasets consisting only in data paths.
using UnlabeledData = std::string;
using UnlabeledDataSet = std::vector<UnlabeledData>;

/// Function to read an unlabeled data set as an array of paths. The
/// \p dataSetFile is the path of the dataset description text file which
/// contains on each line a file name. The rest of the line content apart
/// from what is needed is ignored. An example might look like this (space
/// separated):
///   image0.png
///   image1.png this will be ignored
///   ...............................
/// Another example might look like this (comma separated):
///   image0.png,
///   image1.png,this,will,be,ignored,
///   ...............................
/// The file names are concatenated (prepended) with a common directory path
/// given by \p dataSetDirPath. This function validates that all the data paths
/// are valid and \returns the unlabeled data set.
UnlabeledDataSet readUnlabeledDataSet(llvm::StringRef dataSetFile,
                                      llvm::StringRef dataSetDirPath);

/// Typedefs for labeled datasets consisting in pairs of data paths and integer
/// labels.
using LabeledData = std::pair<std::string, unsigned>;
using LabeledDataSet = std::vector<LabeledData>;

/// Function to read a labeled data set as pairs of {data_path, data_label}.
/// The \p dataSetFile is the path of the dataset description text file which
/// contains on each line a file name and an integer label separated by space
/// or comma. The integer labels start with 0 (0,1,...).  The rest of the line
/// content apart from what is needed is ignored. An example might look like
/// this (space separated):
///   image0.png 0
///   image1.png 13
///   image1.png 15 this will be ignored
///   ..................................
/// Another example might look like this (comma separated):
///   image0.png,0,
///   image1.png,13,
///   image1.png,15,this,will,be,ignored,
///   ..................................
/// The file names are concatenated (prepended) with a common directory path
/// given by \p dataSetDirPath. This function validates that all the data paths
/// are valid and \returns the labeled data set.
LabeledDataSet readLabeledDataSet(llvm::StringRef dataSetFile,
                                  llvm::StringRef dataSetDirPath);

/// Typedefs for time measurement.
using TimeStamp = std::chrono::time_point<std::chrono::high_resolution_clock>;

/// Get current time stamp.
TimeStamp getTimeStamp();

/// Get time duration (seconds) between current moment and a reference starting
/// time stamp.
unsigned getDurationSec(TimeStamp &startTime);

/// Get time duration (seconds, minutes, hours) between current moment and a
/// reference starting time stamp.
void getDuration(TimeStamp &startTime, unsigned &sec, unsigned &min,
                 unsigned &hrs);

} // namespace glow

#endif // GLOW_TOOLS_LOADER_UTILS_H
