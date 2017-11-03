EXEC := buffoncuda
SRCS = $(wildcard *.cu)
OBJS = $(patsubst %.cu, %.o, $(SRCS))

#CXX := /usr/bin/g++-5
CXXFLAGS += -O3
NVCC := nvcc -ccbin $(CXX)

NVCCFLAGS := -m64 -std=c++11 -lineinfo

ALL_CXXFLAGS := $(NVCCFLAGS)
ALL_CXXFLAGS += $(addprefix -Xcompiler ,$(CXXFLAGS))

ALL_LDFLAGS := $(ALL_CXXFLAGS)
ALL_LDFLAGS += $(addprefix -Xlinker ,$(LDFLAGS))

CUDART_DIR := /opt/cuda

# Gencode arguments
SMS ?= 20 30

ifeq ($(GENCODE_FLAGS),)
# Generate SASS code for each SM architecture listed in $(SMS)
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

# Generate PTX code from the highest SM architecture in $(SMS) to guarantee forward-compatibility
HIGHEST_SM := $(lastword $(sort $(SMS)))
ifneq ($(HIGHEST_SM),)
GENCODE_FLAGS += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)
endif
endif

INCLUDES  := -I$(CUDART_DIR)/include -I$(CUDART_DIR)/samples/common/inc 
LIBRARIES := -lcurand

all : build

%.o : %.cu
	$(NVCC) $(INCLUDES) $(ALL_CXXFLAGS) $(GENCODE_FLAGS) -dc $^

#%.o : %.cxx
#	$(NVCC) $(INCLUDES) $(ALL_CXXFLAGS) $(GENCODE_FLAGS) -c $^

build: $(OBJS)
	$(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) $(OBJS) $(LIBRARIES) -o $(EXEC)

.PHONY: clean
clean:
	rm -rf *.o $(EXEC)
