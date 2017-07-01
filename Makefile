
run: noether
	./noether

noether: noether.cpp
	clang++ -std=c++14 -I include/ -Wall noether.cpp -o noether

clean:
	rm -f noether


