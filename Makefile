CC = g++ -std=c++11
OBJ = *.o
EXE = main
FLAGS = -O3 -fopenmp -Wall -Wextra -pedantic -Wno-unused-result -march=native

all:${EXE}

main: main.cpp
	$(CC) -o $@ $^ $(FLAGS) 
clean:
	rm -f $(OBJ) $(EXE)