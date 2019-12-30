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

/// Helper function to trim left and right a string \p str by removing the
/// characters given by \p trimChars.
std::string trimString(std::string str, std::string trimChars = " ");

/// Helper function to split a given string \p str using the given delimiter
/// character \p delim. The function will NOT split the string inside groups
/// of characters which start with one of the caracters specified with
/// \p nopStartChars and end with one of the characters specified with
/// \p nopStopChars. Characters specified with \p ignoreChars will be ignored.
/// This function \returns the parts of the string as a vector of strings.
std::vector<std::string> splitString(std::string str, char delim = ',',
                                     std::string nopStartChars = "(",
                                     std::string nopStopChars = ")",
                                     std::string ignoreChars = " ");

/// Helper class to parse the string description of a generic function call, for
/// example "FUNC('s',1,2.0,"str",[1,2,3],[1.0,2.0,3.0])" where the string is
/// parsed as being a function call with the name "FUNC" having 6 arguments with
/// different types (char,int,float,string,int array,float array) and values.
class FunctionString {
  /// Function name.
  std::string name_;
  /// Function arguments as vector of strings.
  std::vector<std::string> args_;
  /// Split argument into an array of arguments.
  static std::vector<std::string> splitArgArray(std::string arg);

public:
  /// This constructor parses the given string and derives the function name and
  /// arguments.
  FunctionString(std::string str);

  /// Getter for function name.
  std::string getName() const { return name_; }

  /// Getter for function arguments.
  std::vector<std::string> getArgs() const { return args_; }

  /// Get number of function arguments.
  size_t getNumArgs() const { return args_.size(); }

  /// Verify if function has argument with given position.
  bool hasArg(int pos = 0) const;

  /// Get an argument from the argument list as string without
  /// interpretting or stripping any characters.
  std::string getArg(int pos = 0) const;

  /// Get a character argument (delimited with ' or ").
  char getArgChar(int pos = 0) const;

  /// Get a string argument (delimited with ' or ").
  std::string getArgStr(int pos = 0) const;

  /// Get a signed integer argument ("1").
  int getArgInt(int pos = 0) const;

  /// Get a float argument ("2.0").
  float getArgFloat(int pos = 0) const;

  /// Get a signed integer array argument ("[1,2,3]").
  std::vector<int> getArgIntArray(int pos = 0) const;

  /// Get a float array argument ("[1.0,2.0,3.0]").
  std::vector<float> getArgFloatArray(int pos = 0) const;
};

} // namespace glow

#endif // GLOW_TOOLS_LOADER_UTILS_H
