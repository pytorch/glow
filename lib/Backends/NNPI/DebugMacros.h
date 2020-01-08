/*
 * Copyright (c) Glow Contributors. See CONTRIBUTORS file.
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

#ifndef GLOW_NNPI_DEBUG_MACROS_H
#define GLOW_NNPI_DEBUG_MACROS_H

#include "glow/Support/Error.h"
#include "nnpi_inference.h"
#include "nnpi_transformer.h"
#include <chrono>
#include <glog/logging.h>
#include <string>

// Macro for memory instrumentation.
#if NNPI_COLLECT_MEM_USAGE
#include <atomic>
#include <fstream>
#include <iostream>
#include <unistd.h>

static std::atomic<double> dbg_mem_usage_last_vm_value(0.0);
static std::atomic<double> dbg_mem_usage_last_rss_value(0.0);
#define DBG_MEM_USAGE(msg)                                                     \
  {                                                                            \
    double vmSize = 0.0;                                                       \
    double rssSize = 0.0;                                                      \
    unsigned long vmInput;                                                     \
    long rssInput;                                                             \
    {                                                                          \
      std::string notUsed;                                                     \
      std::ifstream ifs("/proc/self/stat", std::ios_base::in);                 \
      for (int i = 0; i < 22 /* unused fields in stat line */; i++)            \
        ifs >> notUsed;                                                        \
      ifs >> vmInput >> rssInput;                                              \
    }                                                                          \
    double pageSize = sysconf(_SC_PAGE_SIZE) / (1024.0 * 1024.0);              \
    vmSize = vmInput / (1024.0 * 1024.0);                                      \
    rssSize = rssInput * pageSize;                                             \
    LOG(INFO) << "[MEM_USAGE][" << __FUNCTION__ << "]" << msg                  \
              << " [MEMORY] vm:" << static_cast<uint64_t>(vmSize)              \
              << " MB RSS:" << static_cast<uint64_t>(rssSize) << " MB "        \
              << "[DELTA] d_vm:"                                               \
              << static_cast<int64_t>(vmInput - dbg_mem_usage_last_vm_value)   \
              << " B D_RSS:"                                                   \
              << static_cast<int64_t>(rssInput - dbg_mem_usage_last_rss_value) \
              << " B \n";                                                      \
    dbg_mem_usage_last_vm_value = vmInput;                                     \
    dbg_mem_usage_last_rss_value = rssInput;                                   \
  }
#warning "####       DBG_MEM_USAGE ENABLED        #####"
#else // Not NNPI_COLLECT_MEM_USAGE
#define DBG_MEM_USAGE(msg)
#endif // NNPI_COLLECT_MEM_USAGE

/// Macro for debug prints.
#ifndef NDEBUG
#include <pthread.h>
#define DBG(msg) LOG(INFO) << "[DEBUG]" << msg << "\n"
#define DBG_TID(msg) DBG("[" << pthread_self() << "]" << msg)
#define ASSERT_WITH_MSG(exp, msg) CHECK(exp) << msg
#else // Not NDEBUG
#define DBG(msg)
#define DBG_TID(msg)
#define ASSERT_WITH_MSG(exp, msg) LOG_ERROR_IF_NOT(exp) << msg
#endif // NDEBUG

static inline std::string GetNNPIErrorDesc(NNPIErrorCode err) {
  NNPIObjectName desc = {0};
  if (nnpiGetNNPIErrorCodeDesc(err, desc) != NNPI_NO_ERROR) {
    return std::string("Failed to get error description");
  }
  return std::string(desc);
}

static inline std::string
GetNNPIInferenceErrorDesc(NNPIInferenceErrorCode err) {
  NNPIObjectName desc = {0};
  if (nnpiGetNNPIInferenceErrorCodeDesc(err, desc) != NNPI_INF_NO_ERROR) {
    return std::string("Failed to get error description");
  }
  return std::string(desc);
}

#define NNPI_ERROR_MSG(res, msg)                                               \
  "NNPIErrorCode [" << res << " = " << GetNNPIErrorDesc(res) << "]: \"" << msg \
                    << "\""

#define NNPI_INF_ERROR_MSG(res, msg)                                           \
  "NNPIInferenceErrorCode [" << res << " = " << GetNNPIInferenceErrorDesc(res) \
                             << "]: \"" << msg << "\""

#define LOG_AND_RETURN_IF(loglevel, exp, msg, retVal)                          \
  {                                                                            \
    if ((exp)) {                                                               \
      LOG(loglevel) << msg;                                                    \
      return retVal;                                                           \
    }                                                                          \
  }

#define LOG_IF_NOT(loglevel, exp) LOG_IF(loglevel, !(exp))

#define LOG_ERROR_IF_NOT(exp) LOG_IF_NOT(ERROR, exp)

#define LOG_AND_RETURN_IF_NOT(loglevel, exp, msg, retVal)                      \
  LOG_AND_RETURN_IF(loglevel, (!(exp)), msg, retVal)

#ifndef NDEBUG
#define ASSERT_LOG_NNPI_ERROR(exp_to_log, msg)                                 \
  {                                                                            \
    NNPIErrorCode exp_res = (exp_to_log);                                      \
    CHECK(exp_res == NNPI_NO_ERROR) << NNPI_ERROR_MSG(exp_res, msg);           \
  }

#else //  NDEBUG
#define ASSERT_LOG_NNPI_ERROR(exp_to_log, msg)                                 \
  {                                                                            \
    NNPIErrorCode exp_res = (exp_to_log);                                      \
    LOG_IF(ERROR, exp_res != NNPI_NO_ERROR) << NNPI_ERROR_MSG(exp_res, msg);   \
  }

#endif //  NDEBUG

#define LOG_NNPI_ERROR(exp_to_log, msg)                                        \
  {                                                                            \
    NNPIErrorCode exp_res = (exp_to_log);                                      \
    LOG_IF(ERROR, exp_res != NNPI_NO_ERROR) << NNPI_ERROR_MSG(exp_res, msg);   \
  }

#define LOG_NNPI_ERROR_RETURN_LLVMERROR(exp, msg)                              \
  {                                                                            \
    NNPIErrorCode exp_res = (exp);                                             \
    LOG_IF(ERROR, exp_res != NNPI_NO_ERROR) << NNPI_ERROR_MSG(exp_res, msg);   \
    if (exp_res != NNPI_NO_ERROR)                                              \
      RETURN_ERR(msg);                                                         \
  }

#define LOG_NNPI_INF_ERROR(exp_to_log, msg)                                    \
  {                                                                            \
    NNPIInferenceErrorCode exp_res = (exp_to_log);                             \
    LOG_IF(ERROR, exp_res != NNPI_INF_NO_ERROR)                                \
        << NNPI_INF_ERROR_MSG(exp_res, msg);                                   \
  }

#define LOG_NNPI_INF_ERROR_RETURN_FALSE(exp, msg)                              \
  {                                                                            \
    NNPIInferenceErrorCode exp_res = (exp);                                    \
    LOG_IF(ERROR, exp_res != NNPI_INF_NO_ERROR)                                \
        << NNPI_INF_ERROR_MSG(exp_res, msg);                                   \
    if (exp_res != NNPI_INF_NO_ERROR)                                          \
      return false;                                                            \
  }

#define LOG_NNPI_INF_ERROR_RETURN_LLVMERROR(exp, msg)                          \
  {                                                                            \
    NNPIInferenceErrorCode exp_res = (exp);                                    \
    LOG_IF(ERROR, exp_res != NNPI_INF_NO_ERROR)                                \
        << NNPI_INF_ERROR_MSG(exp_res, msg);                                   \
    if (exp_res != NNPI_INF_NO_ERROR)                                          \
      RETURN_ERR(msg);                                                         \
  }

#define LOG_NNPI_ERROR_RETURN_VALUE(exp, msg)                                  \
  {                                                                            \
    NNPIErrorCode exp_res = (exp);                                             \
    LOG_IF(ERROR, exp_res != NNPI_NO_ERROR) << NNPI_ERROR_MSG(exp_res, msg);   \
    if (exp_res != NNPI_NO_ERROR)                                              \
      return exp_res;                                                          \
  }

#define LOG_NNPI_ERROR_RETURN_INVALID_HANDLE(exp, msg)                         \
  {                                                                            \
    NNPIErrorCode exp_res = (exp);                                             \
    LOG_IF(ERROR, exp_res != NNPI_NO_ERROR) << NNPI_ERROR_MSG(exp_res, msg);   \
    if (exp_res != NNPI_NO_ERROR)                                              \
      return NNPI_INVALID_NNPIHANDLE;                                          \
  }

#define LOG_NNPI_ERROR_RETURN_FALSE(exp, msg)                                  \
  {                                                                            \
    NNPIErrorCode exp_res = (exp);                                             \
    LOG_IF(ERROR, exp_res != NNPI_NO_ERROR) << NNPI_ERROR_MSG(exp_res, msg);   \
    if (exp_res != NNPI_NO_ERROR)                                              \
      return false;                                                            \
  }

#define LOG_INVALID_HANDLE_RETURN_LLVMERROR(exp, msg)                          \
  {                                                                            \
    NNPIHandle exp_res = (exp);                                                \
    LOG_IF(ERROR, exp_res == NNPI_INVALID_NNPIHANDLE) << msg;                  \
    if (exp_res == NNPI_INVALID_NNPIHANDLE)                                    \
      RETURN_ERR(msg);                                                         \
  }

#define LOG_IF_NOT_RETURN_LLVMERROR(exp, msg)                                  \
  {                                                                            \
    bool exp_res = (exp);                                                      \
    LOG_IF(ERROR, !exp_res) << msg;                                            \
    if (!exp_res)                                                              \
      RETURN_ERR(msg);                                                         \
  }

#define LOG_AND_FAIL_EXECUTE_CALLBACK_IF_NOT(loglevel, exp, msg, runId, ctx,   \
                                             callback)                         \
  {                                                                            \
    bool exp_res = (exp);                                                      \
    LOG_IF(loglevel, !exp_res) << msg;                                         \
    if (!exp_res) {                                                            \
      callback(runId, MAKE_ERR(msg), std::move(ctx));                          \
      return;                                                                  \
    }                                                                          \
  }

#define LOG_AND_FAIL_CALLBACK_IF_NOT(exp, msg, callback)                       \
  {                                                                            \
    bool exp_res = (exp);                                                      \
    LOG_IF(ERROR, !exp_res) << msg;                                            \
    if (!exp_res) {                                                            \
      callback(MAKE_ERR(msg));                                                 \
      return;                                                                  \
    }                                                                          \
  }

#define LOG_AND_CALLBACK_EXECUTE_NNPI_INF_ERROR(exp, msg, runId, ctx,          \
                                                callback)                      \
  {                                                                            \
    NNPIInferenceErrorCode exp_res = (exp);                                    \
    bool nnpiOK = (exp_res == NNPI_INF_NO_ERROR);                              \
    LOG_AND_FAIL_EXECUTE_CALLBACK_IF_NOT(ERROR, nnpiOK, msg, runId, ctx,       \
                                         callback);                            \
  }

#define LOG_AND_CALLBACK_EXECUTE_NNPI_ERROR(exp, msg, runId, ctx, callback)    \
  {                                                                            \
    NNPIErrorCode exp_res = (exp);                                             \
    bool nnpiOK = (exp_res == NNPI_NO_ERROR);                                  \
    LOG_AND_FAIL_EXECUTE_CALLBACK_IF_NOT(ERROR, nnpiOK, msg, runId, ctx,       \
                                         callback);                            \
  }

#define LOG_AND_CALLBACK_NNPI_INF_ERROR(exp, msg, callback)                    \
  {                                                                            \
    NNPIInferenceErrorCode exp_res = (exp);                                    \
    LOG_IF(ERROR, exp_res != NNPI_INF_NO_ERROR)                                \
        << NNPI_INF_ERROR_MSG(exp_res, msg);                                   \
    if (exp_res != NNPI_INF_NO_ERROR)                                          \
      callback(MAKE_ERR(ErrorValue::ErrorCode::RUNTIME_ERROR, msg));           \
  }

// Used in debugging.
#define NNPI_TIMER_START(timer_name)                                           \
  auto timer_name = std::chrono::high_resolution_clock::now();
#define NNPI_TIMER_STOP(timer_name, msg_prefix_)                               \
  {                                                                            \
    auto timer_end_ = std::chrono::high_resolution_clock::now();               \
    std::cout                                                                  \
        << std::string(msg_prefix_) +                                          \
               std::to_string(                                                 \
                   std::chrono::duration_cast<std::chrono::microseconds>(      \
                       timer_end - timer_name)                                 \
                       .count()) +                                             \
               "\n";                                                           \
  }

#endif // GLOW_NNPI_DEBUG_MACROS_H
