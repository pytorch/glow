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

#include "glow/Support/ZipUtils.h"
#include "glow/Support/Memory.h"

#include "llvm/ADT/STLExtras.h"

#include <sstream>

namespace glow {

namespace {
constexpr int MZ_ZIP_LOCAL_DIR_HEADER_SIZE = 30;
constexpr uint64_t kFieldAlignment = 64;

static std::string getPadding(size_t cursor, const std::string &filename,
                              size_t size) {
  size_t start = cursor + MZ_ZIP_LOCAL_DIR_HEADER_SIZE + filename.size() +
                 sizeof(mz_uint16) * 2;
  if (size >= MZ_UINT32_MAX || cursor >= MZ_UINT32_MAX) {
    start += sizeof(mz_uint16) * 2;
    if (size >= MZ_UINT32_MAX) {
      start += 2 * sizeof(mz_uint64);
    }
    if (cursor >= MZ_UINT32_MAX) {
      start += sizeof(mz_uint64);
    }
  }
  size_t mod = start % kFieldAlignment;
  size_t next_offset = (mod == 0) ? start : (start + kFieldAlignment - mod);
  size_t padding_size = next_offset - start;
  std::string buf(padding_size + 4, 'Z');
  // zip extra encoding (key, size_of_extra_bytes)
  buf[0] = 'F';
  buf[1] = 'B';
  buf[2] = (uint8_t)padding_size;
  buf[3] = (uint8_t)(padding_size >> 8);
  return buf;
}
} // namespace

size_t istreamReadFunc(void *pOpaque, mz_uint64 file_ofs, void *pBuf,
                       size_t n) {
  auto self = static_cast<ZipReader *>(pOpaque);
  return self->read(file_ofs, static_cast<char *>(pBuf), n);
}

ZipReader::ZipReader(const std::string &file_name)
    : ar_(glow::make_unique<mz_zip_archive>()),
      in_(glow::make_unique<FileAdapter>(file_name)) {
  init();
}

ZipReader::~ZipReader() {
  mz_zip_reader_end(ar_.get());
  valid("closing reader for archive ", archive_name_.c_str());
}

void ZipReader::init() {
  assert(in_ != nullptr);
  assert(ar_ != nullptr);
  memset(ar_.get(), 0, sizeof(mz_zip_archive));
  size_t size = in_->size();
  ar_->m_pIO_opaque = this;
  ar_->m_pRead = istreamReadFunc;
  mz_zip_reader_init(ar_.get(), size, 0);
  valid("reading zip archive");
  // figure out the archive_name (i.e. the zip folder all the other files are
  // in) all lookups to getRecord will be prefixed by this folder
  int n = mz_zip_reader_get_num_files(ar_.get());
  if (n == 0) {
    LOG(FATAL) << "archive does not contain any files";
  }
  size_t name_size = mz_zip_reader_get_filename(ar_.get(), 0, nullptr, 0);
  valid("getting filename");
  std::string buf(name_size, '\0');
  mz_zip_reader_get_filename(ar_.get(), 0, &buf[0], name_size);
  valid("getting filename");
  auto pos = buf.find_first_of('/');
  if (pos == std::string::npos) {
    LOG(FATAL) << "file in archive is not in a subdirectory";
  }
  archive_name_ = buf.substr(0, pos);
}

size_t ZipReader::getRecordID(const std::string &name) {
  std::stringstream ss;
  ss << archive_name_ << "/" << name;
  size_t result =
      mz_zip_reader_locate_file(ar_.get(), ss.str().c_str(), nullptr, 0);
  if (ar_->m_last_error == MZ_ZIP_FILE_NOT_FOUND) {
    LOG(FATAL) << "file not found: " << ss.str();
  }
  valid("locating file ", name.c_str());
  return result;
}

bool ZipReader::hasRecord(const std::string &name) {
  std::stringstream ss;
  ss << archive_name_ << "/" << name;
  mz_zip_reader_locate_file(ar_.get(), ss.str().c_str(), nullptr, 0);
  bool result = ar_->m_last_error != MZ_ZIP_FILE_NOT_FOUND;
  if (!result) {
    ar_->m_last_error = MZ_ZIP_NO_ERROR;
  }
  valid("attempting to locate file ", name.c_str());
  return result;
}

std::string ZipReader::getRecord(const std::string &name) {
  size_t key = getRecordID(name);
  mz_zip_archive_file_stat stat;
  mz_zip_reader_file_stat(ar_.get(), key, &stat);
  valid("retrieving file meta-data for ", name.c_str());
  std::string data;
  data.resize(stat.m_uncomp_size);
  mz_zip_reader_extract_to_mem(ar_.get(), key, &data[0], stat.m_uncomp_size, 0);
  valid("reading file ", name.c_str());
  return data;
}

void ZipReader::valid(const char *what, const char *info) {
  auto err = mz_zip_get_last_error(ar_.get());
  if (err != MZ_ZIP_NO_ERROR) {
    LOG(FATAL) << "PytorchStreamReader failed " << what << info << ": "
               << mz_zip_get_error_string(err);
  }
}

size_t ostreamWriteFunc(void *pOpaque, mz_uint64 file_ofs, const void *pBuf,
                        size_t n) {
  auto *self = static_cast<ZipWriter *>(pOpaque);
  if (self->current_pos_ != file_ofs) {
    // xxx - windows ostringstream refuses to seek to the end of an empty string
    // so we workaround this by not calling seek unless necessary
    // in the case of the first write (to the empty string) file_ofs and
    // current_pos_ will be 0 and the seek won't occur.
    self->out_->seekp(file_ofs);
    if (!*self->out_) {
      return 0;
    }
  }
  self->out_->write(static_cast<const char *>(pBuf), n);
  if (!*self->out_) {
    return 0;
  }
  self->current_pos_ = file_ofs + n;
  return n;
}

ZipWriter::ZipWriter(std::ostream *out, const std::string &archive_name)
    : out_(out), finalized_{false}, ar_(glow::make_unique<mz_zip_archive>()),
      archive_name_(archive_name) {
  memset(ar_.get(), 0, sizeof(mz_zip_archive));
  ar_->m_pIO_opaque = this;
  ar_->m_pWrite = ostreamWriteFunc;
  mz_zip_writer_init_v2(ar_.get(), 0, MZ_ZIP_FLAG_WRITE_ZIP64);
}

void ZipWriter::writeRecord(const std::string &name, const void *data,
                            size_t size, bool compress) {
  assert(!finalized_);
  std::stringstream ss;
  ss << archive_name_ << "/" << name;
  const std::string full_name = ss.str();
  std::string padding = getPadding(ar_->m_archive_size, full_name, size);
  uint32_t flags = compress ? MZ_BEST_COMPRESSION : 0;
  mz_zip_writer_add_mem_ex_v2(ar_.get(), full_name.c_str(), data, size, nullptr,
                              0, flags, 0, 0, nullptr, padding.c_str(),
                              padding.size(), nullptr, 0);
  valid("writing file ", name.c_str());
}

void ZipWriter::writeEndOfFile() {
  assert(!finalized_);
  finalized_ = true;
  mz_zip_writer_finalize_archive(ar_.get());
  mz_zip_writer_end(ar_.get());
  valid("writing central directory for archive ", archive_name_.c_str());
}

void ZipWriter::valid(const char *what, const char *info) {
  auto err = mz_zip_get_last_error(ar_.get());
  if (err != MZ_ZIP_NO_ERROR) {
    LOG(FATAL) << "ZipWriter failed " << what << info << ": "
               << mz_zip_get_error_string(err);
  }
  if (!*out_) {
    LOG(FATAL) << "ZipWriter failed " << what << info << ".";
  }
}

ZipWriter::~ZipWriter() {
  if (!finalized_) {
    writeEndOfFile();
  }
}
} // namespace glow
