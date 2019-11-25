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

#include "Loader.h"

using namespace glow;

int main(int argc, char **argv) {

  // Parse command line parameters. All the options will be available as part of
  // the loader object.
  parseCommandLine(argc, argv);

  // Initialize the loader object.
  Loader loader;

  // Emit bundle flag should be true.
  CHECK(emittingBundle())
      << "Bundle output directory not provided. Use the -emit-bundle option!";

  // Load the model.
  loader.loadModel();

  // Compile the model with default options.
  CompilationContext cctx = loader.getCompilationContext();
  loader.compile(cctx);

  return 0;
}
