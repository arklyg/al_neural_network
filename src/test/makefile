#preprocess
ifeq ($(MWNEURALNETWORK_SRC_PATH),)
	MWNEURALNETWORK_SRC_PATH = $(shell pwd)/..
endif

include $(MWNEURALNETWORK_SRC_PATH)/makefile.incl

BUILD = testneuralnetwork testreloadneuralnetwork

all:
	for i in $(BUILD); do \
		$(CXX) $(CXXFLAGS) $(MACRO_DEF) $(INC_MWNEURALNETWORK) $(INC_MWMATH) $(INC_MWSTL) $(INC_MWGLOBAL) $(INC_LOG4CPLUS) -c $(addsuffix .cc, $$i) -o $(addsuffix .o, $$i); \
		$(CXX) $(addsuffix .o, $$i) $(STATIC_LIB_MWNEURALNETWORK) $(STATIC_LIB_MWMATH) $(STATIC_LIB_MWSTL) $(STATIC_LIB_MWGLOBAL) $(STATIC_LIB_LOG4CPLUS) $(LIB_RT) $(LIB_PTHREAD) -o $$i; \
	done

clean:
	$(CLEAN)

rebuild:
	$(REBUILD)
