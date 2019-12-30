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

// -----------------------------------------------------------------------------
//                                TIME UTILS
// -----------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------
//                              STRING UTILS
// -----------------------------------------------------------------------------
std::string glow::trimString(std::string str, std::string trimChars) {
  // Trim left.
  while (!str.empty() && (trimChars.find(str.front()) != std::string::npos)) {
    str = str.substr(1);
  }
  // Trim right.
  while (!str.empty() && (trimChars.find(str.back()) != std::string::npos)) {
    str = str.substr(0, str.size() - 1);
  }
  return str;
}

std::vector<std::string> glow::splitString(std::string str, char delim,
                                           std::string nopStartChars,
                                           std::string nopStopChars,
                                           std::string ignoreChars) {
  checkCond(nopStartChars.size() == nopStopChars.size(),
            "Start & stop characters for split no-operation mismatch in size!");
  std::vector<std::string> partArray;
  // Trim string and if empty then return empty array.
  str = trimString(str, ignoreChars);
  if (str.empty()) {
    return partArray;
  }
  // Split.
  std::string part = "";
  auto nopActive = false;
  auto nopCharPos = std::string::npos;
  auto strOld = str;
  auto strNew = str + delim;
  for (const char &ch : strNew) {
    // Activate/deactivate no operation state.
    if (!nopActive) {
      auto charPos = nopStartChars.find(ch);
      if (charPos != std::string::npos) {
        nopActive = true;
        nopCharPos = charPos;
      }
    } else {
      if (ch == nopStopChars[nopCharPos]) {
        nopActive = false;
        nopCharPos = std::string::npos;
      }
    }
    // Split action.
    if (!nopActive) {
      // Ignore characters.
      if (ignoreChars.find(ch) != std::string::npos) {
        continue;
      }
      if (ch == delim) {
        partArray.push_back(part);
        part = "";
      } else {
        part += ch;
      }
    } else {
      part += ch;
    }
  }
  checkCond(nopActive == false,
            "String \"" + strOld + "\" is incomplete! A \"" +
                nopStopChars[nopCharPos] + "\" character is missing!");
  return partArray;
}

FunctionString::FunctionString(std::string str) {
  str = trimString(str, " ");
  auto braceStartPos = str.find('(');
  auto braceStopPos = str.find(')');
  if (braceStartPos == std::string::npos) {
    checkCond(braceStopPos == std::string::npos,
              "Function string: mismatch between opened/closed brackets!");
    // Function without brackets.
    name_ = str;
  } else {
    // Function with brackets.
    checkCond(braceStopPos == str.size() - 1,
              "Function string: with arguments must end with a ')' character!");
    checkCond(braceStartPos != 0, "Function name is empty!");
    name_ = str.substr(0, braceStartPos);
    auto argStr = str.substr(braceStartPos + 1, str.size() - name_.size() - 2);
    args_ = splitString(argStr, ',', "\"['", "\"]'", " ");
  }
  name_ = trimString(name_, " ");
}

bool FunctionString::hasArg(int pos) const {
  return ((0 <= pos) && (pos < (int)args_.size()));
}

std::string FunctionString::getArg(int pos) const {
  checkCond(hasArg(pos), "Function string: invalid argument index!");
  return args_[pos];
}

char FunctionString::getArgChar(int pos) const {
  auto arg = getArg(pos);
  checkCond(arg.size() == 3, "Function string: character argument invalid!");
  checkCond(
      ((arg.front() == '\'') && (arg.back() == '\'')) ||
          ((arg.front() == '"') && (arg.back() == '"')),
      "Function string: character argument should start/end with ' or \"!");
  return arg[1];
}

std::string FunctionString::getArgStr(int pos) const {
  auto arg = getArg(pos);
  checkCond(arg.size() >= 2, "Function string: string argument invalid!");
  checkCond(((arg.front() == '\'') && (arg.back() == '\'')) ||
                ((arg.front() == '"') && (arg.back() == '"')),
            "Function string: string argument should start/end with ' or \"!");
  return arg.substr(1, arg.size() - 2);
}

int FunctionString::getArgInt(int pos) const { return std::stoi(getArg(pos)); }

float FunctionString::getArgFloat(int pos) const {
  return std::stof(getArg(pos));
}

std::vector<std::string> FunctionString::splitArgArray(std::string arg) {
  checkCond(arg.size() > 2,
            "Function string: array argument has invalid size!");
  checkCond(arg.front() == '[',
            "Function string: array argument must start with '['!");
  checkCond(arg.back() == ']',
            "Function string: array argument must end with ']'!");
  return splitString(arg, ',', "", "", " []");
}

std::vector<int> FunctionString::getArgIntArray(int pos) const {
  std::vector<int> argIntArray;
  for (auto item : splitArgArray(getArg(pos))) {
    argIntArray.push_back(std::stoi(item));
  }
  return argIntArray;
}

std::vector<float> FunctionString::getArgFloatArray(int pos) const {
  std::vector<float> argIntArray;
  for (auto item : splitArgArray(getArg(pos))) {
    argIntArray.push_back(std::stof(item));
  }
  return argIntArray;
}
