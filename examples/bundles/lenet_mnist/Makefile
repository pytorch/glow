# The path to the image-classifier executable.
LOADER?=~/src/build/glow/bin/image-classifier

# The root directory of the Glow repo.
GLOW_SRC?=~/src/glow

# Should quantize the network (YES/NO)?
QUANTIZE?=YES

# Path to the images.
IMAGES=${GLOW_SRC}/tests/images/mnist

# Input name of the model.
MODEL_INPUT_NAME=data

# Compiler.
CXX=clang++

run: lenet_mnist
	cd build; \
	for file in ${IMAGES}/*; do \
		./lenet_mnist $$file; \
	done

# Build executable for floating point lenet_mnist.
lenet_mnist: build/main.o build/lenet_mnist.o
	${CXX} -o build/lenet_mnist build/lenet_mnist.o build/main.o -lpng

profile.yml: download_weights
	# Capture quantization profile based on all inputs.
	# Note, Interpreter backend is used to collect the profile data.
	${LOADER} ${IMAGES}/*.png -image_mode=0to1 -dump_profile=profile.yml -m lenet_mnist -model_input_name=${MODEL_INPUT_NAME}

ifeq ($(QUANTIZE),YES)
build/lenet_mnist.o: profile.yml
	mkdir -p build
	# Create bundle with quantized weights and computation graph.
	${LOADER} ${IMAGES}/3_1020.png -image_mode=0to1 -load_profile=profile.yml -m lenet_mnist -model_input_name=${MODEL_INPUT_NAME} -cpu -emit-bundle build -g
else
build/lenet_mnist.o: download_weights
	mkdir -p build
	${LOADER} ${IMAGES}/3_1020.png -image_mode=0to1 -m lenet_mnist -model_input_name=${MODEL_INPUT_NAME} -cpu -emit-bundle build -g
endif

build/main.o: main.cpp
	mkdir -p build
	${CXX} -std=c++11 -c -g main.cpp -o build/main.o

download_weights:
	for file in predict_net.pbtxt predict_net.pb init_net.pb; do \
		wget http://fb-glow-assets.s3.amazonaws.com/models/lenet_mnist/$$file -P lenet_mnist -nc; \
	done

clean:
	rm -rf ./build
	rm -f profile.yml
