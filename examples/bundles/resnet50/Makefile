# Suppress display of executed commands.
$(VERBOSE).SILENT:

# The path to the loader executable.
LOADER?=~/src/build/Glow/bin/loader

# The root directory of the Glow repo.
GLOW_SRC?=~/src/Glow

# Should quantize the network (YES/NO)?
QUANTIZE?=YES

# Path to the images.
IMAGES=${GLOW_SRC}/tests/images/imagenet

# Compiler.
CXX=clang++

run: resnet50
	cd build; \
	for file in ${IMAGES}/*; do \
		./resnet50 $$file; \
	done

# Build executable for floating point resnet50.
resnet50: build/main.o build/resnet50.o
	${CXX} -o build/resnet50 build/resnet50.o build/main.o -lpng

profile.yml: download_weights
	# Capture quantization profile based on all inputs.
	# Note, Interpreter backend is used to collect the profile data.
	${LOADER} ${IMAGES}/*.png -image_mode=0to1 -dump_profile=profile.yml -m resnet50

ifeq ($(QUANTIZE),YES)
build/resnet50.o: profile.yml
	mkdir -p build
	# Create bundle with quantized weights and computation graph.
	${LOADER} ${IMAGES}/dog_207.png -image_mode=0to1 -load_profile=profile.yml -m resnet50 -cpu -emit-bundle build -g
else
build/resnet50.o: download_weights
	mkdir -p build
	${LOADER} ${IMAGES}/dog_207.png -image_mode=0to1 -m resnet50 -cpu -emit-bundle build -g
endif

build/main.o: resnet50.cpp
	mkdir -p build
	${CXX} -std=c++11 -c -g resnet50.cpp -o build/main.o

download_weights:
	for file in predict_net.pbtxt predict_net.pb init_net.pb; do \
		wget http://fb-glow-assets.s3.amazonaws.com/models/resnet50/$$file -P resnet50 -nc; \
	done

clean:
	rm -rf ./build
	rm -f profile.yml
