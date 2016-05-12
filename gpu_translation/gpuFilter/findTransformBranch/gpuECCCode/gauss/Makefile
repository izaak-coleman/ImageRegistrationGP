OBJ=main.o findTransformV3Gauss.o
EXE=gpuECC
CXX=g++
COMPFLAGS=-Wall -ggdb -MMD -I/vol/bitbucket/ic711/usr/local/include -L/vol/bitbucket/ic711/usr/local/lib
LIBS=`pkg-config --libs opencv --cflags opencv`
OBJDIR=./objects/

$(EXE):$(OBJ)
	$(CXX) $(COMPFLAGS) $(OBJ) -o $(EXE) $(LIBS)

%.o: %.cpp
	$(CXX) $(COMPFLAGS) -c $<

-include $(OBJ:.o=.d)	

.PHONY: clean

clean:
	echo cleaning...
	rm ./*.o
	rm ./*.d


