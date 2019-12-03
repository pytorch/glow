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

#Native Build:

##Prerequisites:

###LLVM:

Download LLVM Installation Binary (includes CLANG) from:

http://releases.llvm.org/download.html

Download source: http://releases.llvm.org/7.0.0/llvm-7.0.0.src.tar.xz

###HOWTO with GIT
    git clone --single-branch -b release_70 http://llvm.org/git/llvm.git
    cd llvm/tools/
    git clone --single-branch -b release_70 http://llvm.org/git/clang.git

#####Optional if want to build clang from source.

    cd clang/tools
    git clone --single-branch -b release_70 http://llvm.org/git/clang-tools-extra.git extra


####Building LLVM
Open Visual Studio native x64 command prompt, and navigate to directory where you want your build folder to be.
Commmand example:

    mkdir llvmbuilddir64    
    cd llvmbuilddir64    
    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_GENERATOR_PLATFORM=x64 -DLLVM_INCLUDE_TESTS=OFF -DLLVM_INCLUDE_GO_TESTS=OFF c:\llvm_source\llvm-7.0.0.src

Can be build from VS solution, but can also be done from command line: cmake --build .

####Alternative to building llvm libraries
An alternative way is to download pre-build library binaries from (Debug Symbols files not included):
http://fb-glow-assets.s3.amazonaws.com/bin/llvmbuilddir64.7z

For this method directory structure needs to be:
> c:\projects\llvm\llvm
> 
> c:\projects\llvm\llvmbuilddir64

###GLOW Support libraries
#####Prebuild libraries
Built with Visual Studio 2017
clone https://github.com/ayermolo/glowWindowsSupportLibraries.git

####Building from Scratch
#####GLOG


Clone the source from https://github.com/google/glog
or download from https://github.com/google/glog/releases (glog-0.4.0 is recommended)

create a build folder inside glog and execute following command from inside the build folder to create the project, build and install

    cmake ../ -DCMAKE_GENERATOR_PLATFORM=x64 -DCMAKE_INSTALL_RUNSTATEDIR:PATH="" -DCMAKE_INSTALL_PREFIX:PATH="<Glog_Location>" -DWITH_GFLAGS:BOOL="0"    
    cmake --build . --config Release    
    cmake --build . --config Debug    
    cmake --build . --target install --config Debug    
    cmake --build . --target install --config Release



#####ZLIB/PNG
This repo contains modified cmake files to get it working under windows:

    git clone https://github.com/ayermolo/glowWindowsSupportLibraries.git
    cd glowWindowsSupportLibraries\tools
    mkdir libpngBuild
    cmake -G "Visual Studio 15 2017" -DCMAKE_GENERATOR_PLATFORM=x64 -DPNG_BUILD_ZLIB=ON -DZLIB_INCLUDE_DIR="../zlib-1.2.11" -DPNG_STATIC=OFF -DPNG_SHARED=ON -DCMAKE_INSTALL_PREFIX:PATH=../libpngInstall ..\lpng1636
    cmake --build . --config Debug
    cmake --build . --config Release
    cmake --build . --target install --config Debug
    cmake --build . --target install --config Release

#####Protobuf
Do this in VS terminal opened as admin

    git clone https://github.com/google/protobuf.git
     mkdir protoBufBuild
     cd protoBufBuild
     cmake ..\cmake -DBUILD_SHARED_LIBS=TRUE -Dprotobuf_BUILD_TESTS=OFF -DCMAKE_GENERATOR_PLATFORM=x64 -DCMAKE_INSTALL_PREFIX:PATH=../protobufInstall
    cmake --build . --config Debug
    cmake --build . --config Release
    cmake --build . --target install --config Debug
    cmake --build . --target install --config Release


Open VS solution and build x64 configurations you need.


#####Misc Issues
If you are experiencing linking error "linker ran out of memory". Try modifying C:\Program Files (x86)\Microsoft Visual Studio\2017\Professional\Common7\IDE\VC\VCTargets\Microsoft.Cpp.props

    <PreferredToolArchitecture>x64</PreferredToolArchitecture>
More info if you want to do it per project: https://developercommunity.visualstudio.com/content/problem/160714/memory-error-for-linker-in-vs-155-x64.html


####Glow CMake configuration Command
    cmake ../glow -DCMAKE_GENERATOR_PLATFORM=x64 -DGLOW_WITH_OPENCL=0 -DCMAKE_PREFIX_PATH="C:\projects\llvm\llvmbuilddir64;C:\projects\glowWindowsSupportLibraries\tools\glogInstall;C:\projects\glowWindowsSupportLibraries\tools\libpngInstall;C:\projects\glowWindowsSupportLibraries\tools\protobufInstall"  -Dgtest_force_shared_crt=TRUE  -Dgtest_force_shared_crt=TRUE -DLINK_PROTOBUF_AS_DLL=ON


####Running Glow
    Copy dll files in to {Relase, Debug}  folder where image-classifier.exe is.
    {your_path}\lpng1635\projects\vstudio\x64\{Release, Debug}\libpng16.dll
    {your_path}\protobuf\tmp64\{Release, Debub}\*.dll