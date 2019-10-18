## Build Glow for Windows x86:
------------------------------

This document provides a short description about how to build Glow for Windows
using the MSYS2 MINGW build environment. The instructions below apply for the
**x86_64** build (64 bit). For the **x86** target build (32 bit) the instructions
are similar (replace *mingw64* with *mingw32* in all the instructions below).

### Setup the MSYS2 build environment:
1. Download and install Msys2 from the following location:

   http://repo.msys2.org/distrib/x86_64/msys2-x86_64-20190524.exe

2. Start the Msys2 MinGW64 shell from the windows command prompt.
   All the subsequent commands will be executed in the Msys2 shell.

   ```
   C:\msys64\msys2_shell.cmd -mingw64
   ```

3. Install LLVM 8.0.1 manually (make sure NOT to install the latest available LLVM).
   Do NOT use the command "**pacman -Su**" for updating all the installed packages since 
   Glow requires a specific LLVM version. Execute the commands below one at a time.

   ```
   wget http://repo.msys2.org/mingw/x86_64/mingw-w64-x86_64-llvm-8.0.1-1-any.pkg.tar.xz
   wget http://repo.msys2.org/mingw/x86_64/mingw-w64-x86_64-clang-8.0.1-1-any.pkg.tar.xz
   wget http://repo.msys2.org/mingw/x86_64/mingw-w64-x86_64-polly-8.0.1-1-any.pkg.tar.xz
   pacman -U mingw-w64-x86_64-llvm-8.0.1-1-any.pkg.tar.xz
   pacman -U mingw-w64-x86_64-clang-8.0.1-1-any.pkg.tar.xz
   pacman -U mingw-w64-x86_64-polly-8.0.1-1-any.pkg.tar.xz
   rm mingw-w64-x86_64-llvm-8.0.1-1-any.pkg.tar.xz
   rm mingw-w64-x86_64-clang-8.0.1-1-any.pkg.tar.xz
   rm mingw-w64-x86_64-polly-8.0.1-1-any.pkg.tar.xz
   ```

4. Install the dependencies required for Glow. During package installation it might be
   required to restart the Msys2 shell. Execute the commands below one at a time.

   ```
   pacman -Sy
   pacman -S base-devel zip git mingw-w64-x86_64-toolchain
   pacman -S make mingw-w64-x86_64-cmake mingw-w64-x86_64-protobuf
   pacman -S mingw-w64-x86_64-libpng mingw-w64-x86_64-graphviz mingw-w64-x86_64-python3
   pacman -S mingw-w64-x86_64-glog
   ```

### Build Glow:
1. Restart the Msys2 MinGW64 shell from the windows command prompt:

   ```
   C:\msys64\msys2_shell.cmd -mingw64
   ```

2. Clone the Glow repository:

   ```
   cd /mingw64
   git clone https://github.com/pytorch/glow.git
   ```

3. Update the submodules:

   ```
   cd glow
   git submodule update --init --recursive
   cd ..
   ```

4. Configure Cmake:

   ```
   mkdir glow_build
   cd glow_build
   cmake -G "MSYS Makefiles" -DCMAKE_BUILD_TYPE=Release ../glow/
   ```

5. Build the project:

   ```
   make -j 8
   cd ..
   ```

6. Package Glow (apart from the executables some DLL dependencies are required from the Msys2 environment):

   ```
   mkdir glow_package
   cp glow_build/bin/model-compiler.exe glow_package
   cp glow_build/bin/model-runner.exe glow_package
   cp glow_build/bin/text-translator.exe glow_package
   cp /mingw64/bin/libstdc++-6.dll glow_package
   cp /mingw64/bin/zlib1.dll glow_package
   cp /mingw64/bin/libwinpthread-1.dll glow_package
   cp /mingw64/bin/libgcc_s_seh-1.dll glow_package
   cp /mingw64/bin/libprotobuf.dll glow_package
   cp /mingw64/bin/libpng16-16.dll glow_package
   ```
