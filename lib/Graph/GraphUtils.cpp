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

#include "glow/Graph/Graph.h"
#include "glow/Graph/Node.h"
#include "glow/Graph/Nodes.h"
#include "glow/Graph/Utils.h"
#include "glow/Support/Debug.h"
#include <list>
#include <set>

#define DEBUG_TYPE "graph-utils"

using namespace glow;

/// get total size of the selected axes \p axes from dimensions \p dim.
size_t glow::getDimSizeOfAxes(llvm::ArrayRef<dim_t> dim,
                              llvm::ArrayRef<unsigned_t> axes) {
  size_t axesSize = 1;
  for (const auto axis : axes) {
    axesSize *= dim[axis];
  }
  return axesSize;
}

/// Reshape by combining consecutive \p size axes starting from \p axis.
ShapeVector glow::getNewShapeCombineAxes(llvm::ArrayRef<dim_t> dims,
                                         unsigned_t axis, size_t size) {
  assert(axis + size <= dims.size() &&
         "Cannot remove more dimensions than exist.");
  ShapeVector newDims(dims.begin(), dims.end());

  for (size_t i = size - 1; i > 0; i--) {
    assert(axis + i <= dims.size() &&
           "Axis to remove must fit inside dimensions of the provided dims.");
    newDims[axis] *= newDims[axis + i];
    newDims.erase(newDims.begin() + axis + i);
  }
  return newDims;
}

/// \returns a ShapeVector of rank axes.size() less than the input \p dims,
/// where the provided \p axes dimensions are removed from the shape.
ShapeVector glow::getNewShapeWithoutAxes(llvm::ArrayRef<dim_t> dims,
                                         llvm::ArrayRef<unsigned_t> axes) {
  assert(axes.size() <= dims.size() &&
         "Cannot remove more dimensions than exist.");
  ShapeVector newDims(dims.begin(), dims.end());
  ShapeVector shapeAxes(axes.begin(), axes.end());

  // Sort so that looping erase below doesn't fail.
  std::sort(shapeAxes.rbegin(), shapeAxes.rend());

  for (const auto &axis : shapeAxes) {
    assert(axis <= dims.size() &&
           "Axis to remove must fit inside dimensions of the provided dims.");
    newDims.erase(newDims.begin() + axis);
  }
  return newDims;
}
