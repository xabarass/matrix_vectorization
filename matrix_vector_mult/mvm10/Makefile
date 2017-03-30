CC=gcc
CFLAGS=-O3 -mavx -std=c99
VERIFY=-DVERIFY

all: mvm10

mvm10: mvm10.c rdtsc.h utils.h
	$(CC) $(CFLAGS) $(VERIFY) $< -o $@

clean:
	rm mvm10
