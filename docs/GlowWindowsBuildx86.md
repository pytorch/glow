## Build Glow for Windows x86:
------------------------------
# Native Build:

## Prerequisites:

### LLVM:

Download LLVM Installation Binary (includes CLANG) from:

http://releases.llvm.org/download.html

Download source: http://releases.llvm.org/7.0.0/llvm-7.0.0.src.tar.xz

### HOWTO with GIT
    git clone --single-branch -b release_70 http://llvm.org/git/llvm.git
    cd llvm/tools/
    git clone --single-branch -b release_70 http://llvm.org/git/clang.git

##### Optional if want to build clang from source.

    cd clang/tools
    git clone --single-branch -b release_70 http://llvm.org/git/clang-tools-extra.git extra

#### Building LLVM
Open Visual Studio native x64 command prompt, and navigate to directory where you want your build folder to be.
Commmand example:

    mkdir llvmbuilddir64    
    cd llvmbuilddir64    
    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_GENERATOR_PLATFORM=x64 -DLLVM_INCLUDE_TESTS=OFF -DLLVM_INCLUDE_GO_TESTS=OFF c:\llvm_source\llvm-7.0.0.src

Can be build from VS solution, but can also be done from command line: cmake --build .

#### Alternative to building llvm libraries
An alternative way is to download pre-build library binaries from (Debug Symbols files not included):
http://fb-glow-assets.s3.amazonaws.com/bin/llvmbuilddir64.7z

For this method directory structure needs to be:
> c:\projects\llvm\llvm
> 
> c:\projects\llvm\llvmbuilddir64

### GLOW Support libraries
##### Prebuild libraries
Built with Visual Studio 2017
clone https://github.com/ayermolo/glowWindowsSupportLibraries.git

#### Building from Scratch
##### GLOG
Clone the source from https://github.com/google/glog
or download from https://github.com/google/glog/releases (glog-0.4.0 is recommended)

create a build folder inside glog and execute following command from inside the build folder to create the project, build and install

    cmake ../ -DCMAKE_GENERATOR_PLATFORM=x64 -DCMAKE_INSTALL_RUNSTATEDIR:PATH="" -DCMAKE_INSTALL_PREFIX:PATH="<Glog_Location>" -DWITH_GFLAGS:BOOL="0"    
    cmake --build . --config Release    
    cmake --build . --config Debug    
    cmake --build . --target install --config Debug    
    cmake --build . --target install --config Release



##### ZLIB/PNG
This repo contains modified cmake files to get it working under windows:

    git clone https://github.com/ayermolo/glowWindowsSupportLibraries.git
    cd glowWindowsSupportLibraries\tools
    mkdir libpngBuild
    cmake -G "Visual Studio 15 2017" -DCMAKE_GENERATOR_PLATFORM=x64 -DPNG_BUILD_ZLIB=ON -DZLIB_INCLUDE_DIR="../zlib-1.2.11" -DPNG_STATIC=OFF -DPNG_SHARED=ON -DCMAKE_INSTALL_PREFIX:PATH=../libpngInstall ..\lpng1636
    cmake --build . --config Debug
    cmake --build . --config Release
    cmake --build . --target install --config Debug
    cmake --build . --target install --config Release

##### Protobuf
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

##### FOLLY
Need to get and install VCPKG first

    git clone https://github.com/Microsoft/vcpkg.git
    cd vcpckg
    .\bootstrap-vcpkg.bat

Get and build folly and other dependencies.
At the time of this writing vcpkg does not list fmt as dependency for latest Folly, so need to fetch it manually.

    .\vcpkg.exe install fmt:x64-windows
    .\vcpkg.exe install jemalloc:x64-windows
    .\vcpkg.exe install folly:x64-windows --head


##### Misc Issues
If you are experiencing linking error "linker ran out of memory". Try modifying C:\Program Files (x86)\Microsoft Visual Studio\2017\Professional\Common7\IDE\VC\VCTargets\Microsoft.Cpp.props

    <PreferredToolArchitecture>x64</PreferredToolArchitecture>
More info if you want to do it per project: https://developercommunity.visualstudio.com/content/problem/160714/memory-error-for-linker-in-vs-155-x64.html


#### Glow CMake configuration Command
    cmake ../glow -DCMAKE_GENERATOR_PLATFORM=x64 -DGLOW_WITH_OPENCL=0 -DCMAKE_PREFIX_PATH="C:\projects\llvm\llvmbuilddir64;C:\projects\glowWindowsSupportLibraries\tools\glogInstall;C:\projects\glowWindowsSupportLibraries\tools\libpngInstall;C:\projects\glowWindowsSupportLibraries\tools\protobufInstall;C:\projects\gitlab\vcpkg\installed\x64-windows"  -Dgtest_force_shared_crt=TRUE  -Dgtest_force_shared_crt=TRUE -DLINK_PROTOBUF_AS_DLL=ON


#### Running Glow
    Copy dll files in to {Relase, Debug}  folder where image-classifier.exe is.
    {your_path}\lpng1635\projects\vstudio\x64\{Release, Debug}\libpng16.dll
    {your_path}\protobuf\tmp64\{Release, Debub}\*.dll