# vim: set filetype=make nowrap:
# Makefile
#

## Directories
PWD   = $(CURDIR)
I_DIR = ${PWD}/include
S_DIR = ${PWD}/src
L_DIR = ${PWD}/lib
O_DIR = ${PWD}/obj
B_DIR = ${PWD}/bin


## Compiler & linker opts.
CC           = clang++ #clang++, g++
CCSTANDARD   = c++11 # {c++{98,03,11,0x,14,1z},gnu++{98,03,11,0x,14,1z}}
OPTIMIZATION = 3
CCFLAGS      = --pedantic -Wall -Werror -Wshadow -std=${CCSTANDARD} -I ${I_DIR}
LDFLAGS      = -lm -L ${L_DIR}

# Use `make DEBUG=1` to add debugging information, symbol table, etc.
DEBUG ?= 0
ifeq ($(DEBUG), 1)
	CCFLAGS += -DDEBUG -g -ggdb -O0
else
	CCFLAGS += -DNDEBUG -O${OPTIMIZATION}
endif


## Makefile opts.
SHELL = /bin/sh
.SUFFIXES:
.SUFFIXES: .hh .cc .h .c .o


## Files options
TARGET = ${B_DIR}/main
OBJS = $(patsubst ${S_DIR}/%.cc, ${O_DIR}/%.o, $(wildcard ${S_DIR}/*.cc))
RUN_ARGS =

## Linkage
${TARGET}: ${OBJS}
	${CC} ${LDFLAGS} -o $@ $^


## Compilation
${O_DIR}/%.o: ${S_DIR}/%.cc
	${CC} ${CCFLAGS} -c -o $@ $<


## Make options
.PHONY: clean clean-obj clean-all

all:
	make ${TARGET}

clean-obj:
	@rm --force ${OBJS}

clean-bin:
	@rm --force ${TARGET}

clean:
	make clean-obj
	make clean-bin

clean-all:
	make clean

debug:
	make hard DEBUG=1

hard:
	make clean
	make all

run:
	${TARGET} ${RUN_ARGS}

run-hard:
	make clean-obj
	make all
	make run

hard-run:
	make run-hard

help:
	@echo "Type:"
	@echo "  'make all'......................... Build project"
	@echo "  'make run'................ Run binary (if exists)"
	@echo "  'make clean-obj'.............. Clean object files"
	@echo "  'make clean'....... Clean binary and object files"
	@echo "  'make debug'................Compile in DEBUG mode"
	@echo "  'make hard'...................... Clean and build"
	@echo ""
	@echo " Binary will be placed in '${TARGET}'"

