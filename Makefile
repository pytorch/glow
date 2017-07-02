
run: noether
	./build/noether

noether: noether.cpp
	mkdir -p build
	clang++ -fsanitize=address -fno-omit-frame-pointer -g -std=c++14 -I include/ -Wall noether.cpp -o ./build/noether -lpng

clean:
	rm -rf ./build/*


