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
#include <glog/logging.h>

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
    CHECK(exp_res == NNPI_NO_ERROR)                                            \
        << " NNPIErrorCode:" << exp_res << " :" << msg;                        \
  }

#else //  NDEBUG
#define ASSERT_LOG_NNPI_ERROR(exp_to_log, msg)                                 \
  {                                                                            \
    NNPIErrorCode exp_res = (exp_to_log);                                      \
    LOG_IF(ERROR, exp_res != NNPI_NO_ERROR)                                    \
        << "NNPIErrorCode:" << exp_res << " :" << msg;                         \
  }

#endif //  NDEBUG

#define LOG_NNPI_ERROR(exp_to_log, msg)                                        \
  {                                                                            \
    NNPIErrorCode exp_res = (exp_to_log);                                      \
    LOG_IF(ERROR, exp_res != NNPI_NO_ERROR)                                    \
        << "NNPIErrorCode:" << exp_res << " :" << msg;                         \
  }

#define LOG_NNPI_ERROR_RETURN_LLVMERROR(exp, msg)                              \
  {                                                                            \
    NNPIErrorCode res = (exp);                                                 \
    LOG_IF(ERROR, res != NNPI_NO_ERROR)                                        \
        << "NNPIErrorCode:" << res << " :" << msg;                             \
    if (res != NNPI_NO_ERROR)                                                  \
      RETURN_ERR(msg);                                                         \
  }

#define LOG_NNPI_INF_ERROR(exp_to_log, msg)                                    \
  {                                                                            \
    NNPIInferenceErrorCode exp_res = (exp_to_log);                             \
    LOG_IF(ERROR, exp_res != NNPI_INF_NO_ERROR)                                \
        << "NNPIInferenceErrorCode:" << exp_res << " :" << msg;                \
  }

#define LOG_NNPI_INF_ERROR_RETURN_FALSE(exp, msg)                              \
  {                                                                            \
    NNPIInferenceErrorCode res = (exp);                                        \
    LOG_IF(ERROR, res != NNPI_INF_NO_ERROR)                                    \
        << "NNPIInferenceErrorCode:" << res << " :" << msg;                    \
    if (res != NNPI_INF_NO_ERROR)                                              \
      return false;                                                            \
  }

#define LOG_NNPI_INF_ERROR_RETURN_LLVMERROR(exp, msg)                          \
  {                                                                            \
    NNPIInferenceErrorCode res = (exp);                                        \
    LOG_IF(ERROR, res != NNPI_INF_NO_ERROR)                                    \
        << "NNPIInferenceErrorCode:" << res << " :" << msg;                    \
    if (res != NNPI_INF_NO_ERROR)                                              \
      RETURN_ERR(msg);                                                         \
  }

#define LOG_NNPI_ERROR_RETURN_VALUE(exp, msg)                                  \
  {                                                                            \
    NNPIErrorCode res = (exp);                                                 \
    LOG_IF(ERROR, res != NNPI_NO_ERROR)                                        \
        << "NNPIErrorCode:" << res << " :" << msg;                             \
    if (res != NNPI_NO_ERROR)                                                  \
      return res;                                                              \
  }

#define LOG_NNPI_ERROR_RETURN_INVALID_HANDLE(exp, msg)                         \
  {                                                                            \
    NNPIErrorCode res = (exp);                                                 \
    LOG_IF(ERROR, res != NNPI_NO_ERROR)                                        \
        << "NNPIErrorCode:" << res << " :" << msg;                             \
    if (res != NNPI_NO_ERROR)                                                  \
      return NNPI_INVALID_NNPIHANDLE;                                          \
  }

#define LOG_NNPI_ERROR_RETURN_FALSE(exp, msg)                                  \
  {                                                                            \
    NNPIErrorCode res = (exp);                                                 \
    LOG_IF(ERROR, res != NNPI_NO_ERROR)                                        \
        << "NNPIErrorCode:" << res << " :" << msg;                             \
    if (res != NNPI_NO_ERROR)                                                  \
      return false;                                                            \
  }

#define LOG_INVALID_HANDLE_RETURN_LLVMERROR(exp, msg)                          \
  {                                                                            \
    NNPIHandle res = (exp);                                                    \
    LOG_IF(ERROR, res == NNPI_INVALID_NNPIHANDLE) << msg;                      \
    if (res == NNPI_INVALID_NNPIHANDLE)                                        \
      RETURN_ERR(msg);                                                         \
  }

#define LOG_IF_NOT_RETURN_LLVMERROR(exp, msg)                                  \
  {                                                                            \
    bool res = (exp);                                                          \
    LOG_IF(ERROR, !res) << msg;                                                \
    if (!res)                                                                  \
      RETURN_ERR(msg);                                                         \
  }

#define LOG_AND_FAIL_CALLBACK_IF_NOT(loglevel, exp, msg, runId, ctx, callback) \
  {                                                                            \
    bool res = (exp);                                                          \
    LOG_IF(loglevel, !res) << msg;                                             \
    if (!res) {                                                                \
      callback(runId, MAKE_ERR(msg), std::move(ctx));                          \
      return;                                                                  \
    }                                                                          \
  }

#define LOG_AND_CALLBACK_NNPI_INF_ERROR(exp, msg, runId, ctx, callback)        \
  {                                                                            \
    NNPIInferenceErrorCode res = (exp);                                        \
    bool nnpiOK = (res == NNPI_INF_NO_ERROR);                                  \
    LOG_AND_FAIL_CALLBACK_IF_NOT(ERROR, nnpiOK, msg, runId, ctx, callback);    \
  }

#define LOG_AND_CALLBACK_NNPI_ERROR(exp, msg, runId, ctx, callback)            \
  {                                                                            \
    NNPIErrorCode res = (exp);                                                 \
    bool nnpiOK = (res == NNPI_NO_ERROR);                                      \
    LOG_AND_FAIL_CALLBACK_IF_NOT(ERROR, nnpiOK, msg, runId, ctx, callback);    \
  }

#endif // GLOW_NNPI_DEBUG_MACROS_H
