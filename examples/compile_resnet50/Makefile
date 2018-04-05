# Suppress display of executed commands.
$(VERBOSE).SILENT:

# The path to the loader executable.
LOADER?=~/src/build/Glow/bin/loader

# The root directory of the Glow repo.
GLOW_SRC?=~/src/Glow

# Path to the images.
IMAGES=${GLOW_SRC}/tests/images/imagenet

# Compiler.
CXX=clang++

run_resnet50: resnet50
	cd build; \
	for file in ${IMAGES}/*; do \
		./resnet50 $$file; \
	done

run_resnet50_quantized: resnet50_quantized
	cd build_quantized; \
	for file in ${IMAGES}/*; do \
		./resnet50 $$file; \
	done

# Build executable for floating point resnet50.
resnet50: build/main.o build/resnet50.o
	${CXX} -o build/resnet50 build/resnet50.o build/main.o -lpng

# Build executable for quantized resnet50. 
resnet50_quantized: build/main.o build_quantized/resnet50.o
	${CXX} -o build_quantized/resnet50 build_quantized/resnet50.o build/main.o -lpng

build/resnet50.o: download_weights
	mkdir -p build
	${LOADER} ${IMAGES}/dog_207.png -image_mode=0to1 -d resnet50 -cpu -emit-bundle build

build_quantized/resnet50.o: download_weights
	mkdir -p build_quantized
	# Capture quantization profile based on all inputs.
	${LOADER} ${IMAGES}/*.png -image_mode=0to1 -dump_profile=profile.yml -d resnet50
	# Create bundle with quantized weights and computation graph.
	${LOADER} ${IMAGES}/dog_207.png -image_mode=0to1 -load_profile=profile.yml -d resnet50 -cpu -emit-bundle build_quantized

build/main.o:
	mkdir -p build
	${CXX} -std=c++11 -c -g resnet50_standalone.cpp -o build/main.o

download_weights:
	for file in predict_net.pbtxt predict_net.pb init_net.pb; do \
		wget http://fb-glow-assets.s3.amazonaws.com/models/resnet50/$$file -P resnet50 -nc; \
	done

clean:
	rm -rf ./build
	rm -rf ./build_quantized
	rm -f profile.yml
