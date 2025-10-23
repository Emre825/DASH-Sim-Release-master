CC := nvcc
SRC := main.cpp gpu_relu.cu gpu_gemm.cu
TARGET := main
INCDIR := .include

CUDA_LIB_DIR := /usr/local/cuda-11.4/lib64/libcublas.so
CUDA_INC_DIR := /usr/local/cuda-11.4/include/

INCDIR := $(CUDA_INC_DIR)
LDFLAGS += -L$(CUDA_LIB_DIR)
LDLIBS += -lcublas

.PHONY: all clean
all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) -o $@ $(SRC) -I$(INCDIR) $(LDFLAGS) $(LDLIBS)

clean:
	rm -f $(TARGET) *.o