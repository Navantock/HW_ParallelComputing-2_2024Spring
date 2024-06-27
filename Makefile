CC = g++ -std=c++11
NVCC = nvcc -std=c++11
CC_FLAGS = -O3
NVCC_FLAGS = -lineinfo --expt-extended-lambda -use_fast_math --expt-relaxed-constexpr -res-usage -Xcompiler -O3

SRC_DIR = ./src

OBJ = *.o
EXE = serial_ccl cuda_ccl

all: ${EXE}

serial_ccl: $(SRC_DIR)/serial_ccl.cpp
	$(CC) -o $@ $^ $(CC_FLAGS)

cuda_ccl: $(SRC_DIR)/cuda_ccl.cu
	$(NVCC) -o $@ $^ $(NVCC_FLAGS)

clean:
	rm -f $(OBJ) $(EXE)
