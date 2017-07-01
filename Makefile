
run: noether
	./noether

noether: noether.cpp
	clang++ -g -std=c++14 -I include/ -Wall noether.cpp -o noether -lpng

clean:
	rm -f noether


