#preprocess
ifeq ($(MWNEURALNETWORK_SRC_PATH),)
	MWNEURALNETWORK_SRC_PATH = $(shell pwd)/..
endif

include $(MWNEURALNETWORK_SRC_PATH)/makefile.incl

.PHONEY : all clean rebuild install test

TARGET = libmwneuralnetwork.a
LIB_INCS = $(wildcard *.h)
MACRO_DEF = #-D_COMPILE_MODE_TEST_=''

all:$(TARGET)

$(TARGET):$(LIB_OBJS)
	$(AR) $@ $^

%.o:%.cpp
	$(CXX) $(CXXFLAGS) $(MACRO_DEF) $(MWNEURALNETWORK_MACRO) $(INC_MWSTL) $(INC_MWMATH) $(INC_MWGLOBAL) $(INC_LOG4CPLUS) -c $< -o $@

clean:
	$(RM) $(TARGET)
	$(RM) $(LIB_OBJS)

rebuild:
	$(MAKE) clean
	$(MAKE)

install:
	$(CP) $(TARGET) $(DIR_MWMATH_LIB)
	$(CP) $(LIB_INCS) $(DIR_MWMATH_INC)

test:
	$(MAKE)
	$(MAKE) install
	$(MAKE) -C $(DIR_TEST)
