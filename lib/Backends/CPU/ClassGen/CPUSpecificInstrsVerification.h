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
#ifdef GLOW_WITH_CPU

void CPUMaxSplatInst::verify() const {
  assert(getSrc()->getType() == getDest()->getType() && "Invalid type");
  assert(getSrc()->dims() == getDest()->dims() && "Invalid shape");
}

void CPUConvDKKC8Inst::verify() const {
  assert(getSrc()->dims()[3] % getGroup() == 0 &&
         "Input channels must be divisible by group.");
  assert(getDest()->dims()[3] % getGroup() == 0 &&
         "Output channels must be divisible by group.");
  assert(getDest()->getElementType() == getSrc()->getElementType() &&
         "Invalid Element Type");
  assert(getDest()->getElementType() == getFilter()->getElementType() &&
         "Invalid Element Type");
  assert(getDest()->getElementType() == getBias()->getElementType() &&
         "Invalid Element Type");
}

#endif // GLOW_WITH_CPU
