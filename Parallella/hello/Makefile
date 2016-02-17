ESDK=$(EPIPHANY_HOME)
ELIBS=$(ESDK)/tools/host/lib
EINCS=$(ESDK)/tools/host/include
ELDF=$(ESDK)/bsps/current/internal.ldf
EXES=main epiphany.srec
OBJS=epiphany.elf

all: $(EXES)


main: host.c data_struct.h
	gcc -O3 host.c -o main -I $(EINCS) -L $(ELIBS) -le-hal -lrt

epiphany.elf: epiphany.c data_struct.h
	e-gcc -O3  -T $(ELDF) epiphany.c -o epiphany.elf -le-lib
#	e-g++ -O3  -T $(ELDF) ep2.cpp -o epiphany.elf -le-lib

epiphany.srec: epiphany.elf
	e-objcopy --srec-forceS3 --output-target srec epiphany.elf epiphany.srec

clean:
	rm $(EXES) $(OBJS)
