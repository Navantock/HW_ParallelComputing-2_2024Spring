CC = mpicxx -std=c++11
OBJ = *.o
EXE = main
FLAGS = -O3 -march=native

all:${EXE}

main: main.cpp
	$(CC) -o $@ $^ $(FLAGS)
clean:
	rm -f $(OBJ) $(EXE)