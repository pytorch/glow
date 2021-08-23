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
#ifndef GLOW_SUPPORT_ZIPUTILS_H
#define GLOW_SUPPORT_ZIPUTILS_H

#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#include <glog/logging.h>

#include "miniz.h"

namespace glow {

class FileAdapter {
public:
  explicit FileAdapter(const std::string &fileName) {
    fileStream_.open(fileName, std::ifstream::in | std::ifstream::binary);
    if (!fileStream_) {
      LOG(ERROR) << "Cannot open file " << fileName;
    }
    istream_ = &fileStream_;
  }
  size_t size() const {
    auto prev_pos = istream_->tellg();
    validate("getting the current position");
    istream_->seekg(0, istream_->end);
    validate("seeking to end");
    auto result = istream_->tellg();
    validate("getting size");
    istream_->seekg(prev_pos);
    validate("seeking to the original position");
    return result;
  }
  size_t read(uint64_t pos, void *buf, size_t n, const char *what = "") const {
    istream_->seekg(pos);
    validate(what);
    istream_->read(static_cast<char *>(buf), n);
    validate(what);
    return n;
  }
  ~FileAdapter() = default;

private:
  std::ifstream fileStream_;
  std::istream *istream_;

  void validate(const char *what) const {
    if (!*istream_) {
      LOG(ERROR) << "istream reader failed: " << what << ".";
    }
  }
};

/// Zip reader
class ZipReader {
  friend size_t istreamReadFunc(void *pOpaque, uint64_t file_ofs, void *pBuf,
                                size_t n);
  std::unique_ptr<mz_zip_archive> ar_;
  std::string archive_name_;
  std::unique_ptr<FileAdapter> in_;
  void valid(const char *what, const char *info = "");
  size_t read(uint64_t pos, char *buf, size_t n) {
    return in_->read(pos, buf, n, "reading file");
  }
  size_t getRecordID(const std::string &name);

public:
  explicit ZipReader(const std::string &file_name);
  ~ZipReader();
  void init();
  std::string getRecord(const std::string &name);
  bool hasRecord(const std::string &name);
};

/// Zip Writer
class ZipWriter {
  std::ostream *out_;
  bool finalized_{false};
  size_t current_pos_{0};
  std::unique_ptr<mz_zip_archive> ar_;
  std::string archive_name_;

public:
  ZipWriter(std::ostream *out, const std::string &archive_name);
  ~ZipWriter();
  void writeRecord(const std::string &name, const void *data, size_t size,
                   bool compress);
  void writeEndOfFile();
  void valid(const char *what, const char *info);
  friend size_t ostreamWriteFunc(void *pOpaque, uint64_t file_ofs,
                                 const void *pBuf, size_t n);
};
} // namespace glow

#endif // GLOW_SUPPORT_ZIPUTILS_H
