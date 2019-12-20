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

#include "LoaderUtils.h"

using namespace glow;

static void checkCond(bool cond, llvm::StringRef errMsg) {
  if (!cond) {
    llvm::errs() << "GLOW ERROR: " << errMsg << "\n";
    std::exit(1);
  }
}

UnlabeledDataSet glow::readUnlabeledDataSet(llvm::StringRef dataSetFile,
                                            llvm::StringRef dataSetDirPath) {
  // Verify the dataset directory path is valid.
  checkCond(llvm::sys::fs::is_directory(dataSetDirPath),
            strFormat("The dataset path '%s' is not a directory!",
                      dataSetDirPath.data()));
  // Parse the dataset file.
  std::ifstream inpFile(dataSetFile);
  checkCond(inpFile.is_open(), strFormat("Cannot open the dataset file '%s'!",
                                         dataSetFile.data()));
  std::string line;
  UnlabeledDataSet dataset;
  while (std::getline(inpFile, line)) {
    std::string dataPath;
    // Replace commas with spaces.
    while (line.find(",") != std::string::npos) {
      line.replace(line.find(","), 1, " ");
    }
    // Read data path.
    std::istringstream lineStream(line);
    checkCond((lineStream >> dataPath).good(),
              strFormat("Failed parsing the unlabeled dataset file '%s'! Check "
                        "the file has the right format!",
                        dataSetFile.data()));
    // Prepend dataset directory path.
    dataPath = std::string(dataSetDirPath) +
               std::string(llvm::sys::path::get_separator()) + dataPath;
    checkCond(llvm::sys::fs::exists(dataPath),
              strFormat("Data path '%s' does not exist!", dataPath.c_str()));
    dataset.emplace_back(dataPath);
  }
  return dataset;
}

LabeledDataSet glow::readLabeledDataSet(llvm::StringRef dataSetFile,
                                        llvm::StringRef dataSetDirPath) {
  // Verify the dataset directory path is valid.
  checkCond(llvm::sys::fs::is_directory(dataSetDirPath),
            strFormat("The dataset path '%s' is not a directory!",
                      dataSetDirPath.data()));
  // Parse the dataset file.
  std::ifstream inpFile(dataSetFile);
  checkCond(inpFile.is_open(), strFormat("Cannot open the dataset file '%s'!",
                                         dataSetFile.data()));
  std::string line;
  LabeledDataSet dataset;
  while (std::getline(inpFile, line)) {
    std::string dataPath;
    unsigned label;
    // Replace commas with spaces.
    while (line.find(",") != std::string::npos) {
      line.replace(line.find(","), 1, " ");
    }
    // Read data path and label.
    std::istringstream lineStream(line);
    checkCond((lineStream >> dataPath >> label).good(),
              strFormat("Failed parsing the labeled dataset file '%s'! Check "
                        "the file has the right format!",
                        dataSetFile.data()));
    // Prepend dataset directory path.
    dataPath = std::string(dataSetDirPath) +
               std::string(llvm::sys::path::get_separator()) + dataPath;
    checkCond(llvm::sys::fs::exists(dataPath),
              strFormat("Data path '%s' does not exist!", dataPath.c_str()));
    dataset.emplace_back(dataPath, label);
  }
  return dataset;
}

TimeStamp glow::getTimeStamp() {
  return std::chrono::high_resolution_clock::now();
}

unsigned glow::getDurationSec(TimeStamp &startTime) {
  TimeStamp currTime = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::seconds>(currTime - startTime)
      .count();
}

void glow::getDuration(TimeStamp &startTime, unsigned &sec, unsigned &min,
                       unsigned &hrs) {
  unsigned totSec = glow::getDurationSec(startTime);
  unsigned totMin = totSec / 60;
  sec = totSec % 60;
  min = totMin % 60;
  hrs = totMin / 60;
}
