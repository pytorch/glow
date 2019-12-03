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
#include "tests/unittests/BackendTestUtils.h"

using namespace glow;

std::set<std::string> glow::backendTestBlacklist = {
    "learnSqrt2Placeholder/0",
    "trainASimpleNetwork/0",
    "simpleRegression/0",
    "learnXor/0",
    "learnLog/0",
    "circle/0",
    "learnSingleValueConcat/0",
    "trainGRU/0",
    "trainLSTM/0",
    "trainSimpleLinearRegression/0",
    "classifyPlayerSport/0",
    "learnSinus/0",
    "nonLinearClassifier/0",
    "convNetForImageRecognition/0",
    "testFindPixelRegression/0",
    "matrixRotationRecognition/0",
    "learnSparseLengthsSumEmbeddings/0",
    "learnSparseLengthsWeightedSumEmbeddings/0",
    "learnSparseLengthsWeightedSumWeights/0",
};
