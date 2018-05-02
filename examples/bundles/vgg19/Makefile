# Suppress display of executed commands.
$(VERBOSE).SILENT:

# The path to the loader executable.
LOADER?=~/src/build/glow/bin/loader

# The root directory of the Glow repo.
GLOW_SRC?=~/src/glow

# Should quantize the network (YES/NO)?
QUANTIZE?=YES

# Path to the images.
IMAGES=${GLOW_SRC}/tests/images/imagenet

# Compiler.
CXX=clang++

run: vgg19
	cd build; \
	for file in ${IMAGES}/*; do \
		./vgg19 $$file; \
	done

# Build executable for floating point vgg19.
vgg19: build/main.o build/vgg19.o
	${CXX} -o build/vgg19 build/vgg19.o build/main.o -lpng

profile.yml: download_weights
	# Capture quantization profile based on all inputs.
	# Note, Interpreter backend is used to collect the profile data.
	${LOADER} ${IMAGES}/*.png -image_mode=128to127 -dump_profile=profile.yml -m vgg19

ifeq ($(QUANTIZE),YES)
build/vgg19.o: profile.yml
	mkdir -p build
	# Create bundle with quantized weights and computation graph.
	${LOADER} ${IMAGES}/dog_207.png -image_mode=128to127 -load_profile=profile.yml -m vgg19 -cpu -emit-bundle build -g
else
build/vgg19.o: download_weights
	mkdir -p build
	${LOADER} ${IMAGES}/dog_207.png -image_mode=128to127 -m vgg19 -cpu -emit-bundle build -g
endif

build/main.o: vgg19.cpp
	mkdir -p build
	${CXX} -std=c++11 -c -g vgg19.cpp -o build/main.o

download_weights:
	for file in predict_net.pbtxt predict_net.pb init_net.pb; do \
		wget http://fb-glow-assets.s3.amazonaws.com/models/vgg19/$$file -P vgg19 -nc; \
	done

clean:
	rm -rf ./build
	rm -f profile.yml
