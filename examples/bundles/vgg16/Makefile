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

run: vgg16
	cd build; \
	for file in ${IMAGES}/*; do \
		./vgg16 $$file; \
	done

# Build executable for floating point vgg16.
vgg16: build/main.o build/vgg16.o
	${CXX} -o build/vgg16 build/vgg16.o build/main.o -lpng

profile.yml: download_weights
	# Capture quantization profile based on all inputs.
	# Note, Interpreter backend is used to collect the profile data.
	${LOADER} ${IMAGES}/*.png -image_mode=0to256 -dump_profile=profile.yml -m vgg16

ifeq ($(QUANTIZE),YES)
build/vgg16.o: profile.yml
	mkdir -p build
	# Create bundle with quantized weights and computation graph.
	${LOADER} ${IMAGES}/dog_207.png -image_mode=0to256 -load_profile=profile.yml -m vgg16 -cpu -emit-bundle build -g
else
build/vgg16.o: download_weights
	mkdir -p build
	${LOADER} ${IMAGES}/dog_207.png -image_mode=0to256 -m vgg16 -cpu -emit-bundle build -g
endif

build/main.o: vgg16.cpp
	mkdir -p build
	${CXX} -std=c++11 -c -g vgg16.cpp -o build/main.o

download_weights:
	for file in predict_net.pbtxt predict_net.pb init_net.pb; do \
		wget http://fb-glow-assets.s3.amazonaws.com/models/vgg16/$$file -P vgg16 -nc; \
	done

clean:
	rm -rf ./build
	rm -f profile.yml
