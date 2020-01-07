/**
 * Copyright (c) 2017-present, Facebook, Inc.
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
#include "tests/unittests/BackendTestUtils.h"

using namespace glow;

std::set<std::string> glow::backendTestBlacklist = {};

struct BlacklistInitializer {
  BlacklistInitializer() {
    backendTestBlacklist = {
        "BatchMatMulTest_Int8/39", "BatchMatMulTest_Int8/43",
        "BatchMatMulTest_Int8/46", "BatchMatMulTest_Int8/51",
        "BatchMatMulTest_Int8/54", "BatchMatMulTest_Int8/59",
        "BatchMatMulTest_Int8/63", "BatchMatMulTest_Int8/67",
        "BatchMatMulTest_Int8/71", "BatchMatMulTest_Int8/75",
        "BatchMatMulTest_Int8/83", "BatchMatMulTest_Int8/87",
        "BatchMatMulTest_Int8/90", "BatchMatMulTest_Int8/91",
        "BatchMatMulTest_Int8/95", "FCTest_Int8/139",
    };

    for (int i = 0; i < 80; i++) {
      backendTestBlacklist.insert("RWQSLWS_Float16_AccumFloat16/" +
                                  std::to_string(i));
    }
    for (int i = 0; i < 80; i++) {
      backendTestBlacklist.insert("FRWQSLWS_Float16_AccumFloat16/" +
                                  std::to_string(i));
    }

    std::set<std::string> emuOnlyTests = {
        "BatchMatMulTest_Float16/2",  "BatchMatMulTest_Float16/3",
        "BatchMatMulTest_Float16/6",  "BatchMatMulTest_Float16/7",
        "BatchMatMulTest_Float16/10", "BatchMatMulTest_Float16/11",
        "BatchMatMulTest_Float16/14", "BatchMatMulTest_Float16/15",
        "BatchMatMulTest_Float16/18", "BatchMatMulTest_Float16/19",
        "BatchMatMulTest_Float16/21", "BatchMatMulTest_Float16/22",
        "BatchMatMulTest_Float16/23", "BatchMatMulTest_Float16/25",
        "BatchMatMulTest_Float16/26", "BatchMatMulTest_Float16/27",
        "BatchMatMulTest_Float16/30", "BatchMatMulTest_Float16/31",
        "BatchMatMulTest_Float16/33", "BatchMatMulTest_Float16/34",
        "BatchMatMulTest_Float16/35", "BatchMatMulTest_Float16/37",
        "BatchMatMulTest_Float16/38", "BatchMatMulTest_Float16/39",
        "BatchMatMulTest_Float16/41", "BatchMatMulTest_Float16/42",
        "BatchMatMulTest_Float16/43", "BatchMatMulTest_Float16/45",
        "BatchMatMulTest_Float16/46", "BatchMatMulTest_Float16/47",
        "BatchMatMulTest_Float16/49", "BatchMatMulTest_Float16/50",
        "BatchMatMulTest_Float16/51", "BatchMatMulTest_Float16/54",
        "BatchMatMulTest_Float16/55", "BatchMatMulTest_Float16/58",
        "BatchMatMulTest_Float16/59", "BatchMatMulTest_Float16/61",
        "BatchMatMulTest_Float16/62", "BatchMatMulTest_Float16/63",
        "BatchMatMulTest_Float16/65", "BatchMatMulTest_Float16/66",
        "BatchMatMulTest_Float16/67", "BatchMatMulTest_Float16/69",
        "BatchMatMulTest_Float16/70", "BatchMatMulTest_Float16/71",
        "BatchMatMulTest_Float16/73", "BatchMatMulTest_Float16/74",
        "BatchMatMulTest_Float16/75", "BatchMatMulTest_Float16/78",
        "BatchMatMulTest_Float16/79", "BatchMatMulTest_Float16/81",
        "BatchMatMulTest_Float16/82", "BatchMatMulTest_Float16/83",
        "BatchMatMulTest_Float16/85", "BatchMatMulTest_Float16/86",
        "BatchMatMulTest_Float16/87", "BatchMatMulTest_Float16/89",
        "BatchMatMulTest_Float16/90", "BatchMatMulTest_Float16/91",
        "BatchMatMulTest_Float16/93", "BatchMatMulTest_Float16/94",
        "BatchMatMulTest_Float16/95", "ConvTest_Float16/3",
        "ConvTest_Float16/7",         "ConvTest_Float16/9",
        "ConvTest_Float16/11",        "FCTest_Float16/14",
        "FCTest_Float16/17",          "FCTest_Float16/18",
        "FCTest_Float16/19",          "FCTest_Float16/21",
        "FCTest_Float16/22",          "FCTest_Float16/23",
        "FCTest_Float16/24",          "FCTest_Float16/26",
        "FCTest_Float16/27",          "FCTest_Float16/28",
        "FCTest_Float16/29",          "FCTest_Float16/30",
        "FCTest_Float16/31",          "FCTest_Float16/32",
        "FCTest_Float16/33",          "FCTest_Float16/34",
        "FCTest_Float16/47",          "FCTest_Float16/48",
        "FCTest_Float16/49",          "FCTest_Float16/51",
        "FCTest_Float16/52",          "FCTest_Float16/53",
        "FCTest_Float16/54",          "FCTest_Float16/55",
        "FCTest_Float16/56",          "FCTest_Float16/57",
        "FCTest_Float16/58",          "FCTest_Float16/59",
        "FCTest_Float16/60",          "FCTest_Float16/61",
        "FCTest_Float16/62",          "FCTest_Float16/63",
        "FCTest_Float16/64",          "FCTest_Float16/65",
        "FCTest_Float16/66",          "FCTest_Float16/67",
        "FCTest_Float16/68",          "FCTest_Float16/69",
        "FCTest_Float16/83",          "FCTest_Float16/84",
        "FCTest_Float16/86",          "FCTest_Float16/87",
        "FCTest_Float16/88",          "FCTest_Float16/89",
        "FCTest_Float16/90",          "FCTest_Float16/91",
        "FCTest_Float16/92",          "FCTest_Float16/93",
        "FCTest_Float16/94",          "FCTest_Float16/95",
        "FCTest_Float16/96",          "FCTest_Float16/97",
        "FCTest_Float16/98",          "FCTest_Float16/99",
        "FCTest_Float16/100",         "FCTest_Float16/101",
        "FCTest_Float16/102",         "FCTest_Float16/103",
        "FCTest_Float16/104",         "FCTest_Float16/116",
        "FCTest_Float16/117",         "FCTest_Float16/118",
        "FCTest_Float16/119",         "FCTest_Float16/121",
        "FCTest_Float16/122",         "FCTest_Float16/123",
        "FCTest_Float16/124",         "FCTest_Float16/125",
        "FCTest_Float16/126",         "FCTest_Float16/127",
        "FCTest_Float16/128",         "FCTest_Float16/129",
        "FCTest_Float16/130",         "FCTest_Float16/131",
        "FCTest_Float16/132",         "FCTest_Float16/133",
        "FCTest_Float16/134",         "FCTest_Float16/135",
        "FCTest_Float16/136",         "FCTest_Float16/137",
        "FCTest_Float16/138",         "FCTest_Float16/139",
    };

    // If USE_INF_API is set, we are running on real hardware, and need
    // to blacklist additional testcases.
    auto useInfAPI = getenv("USE_INF_API");
    if (useInfAPI && !strcmp(useInfAPI, "1")) {
      for (auto testname : emuOnlyTests) {
        backendTestBlacklist.insert(testname);
      }
    }
  }
} blacklistInitializer;
