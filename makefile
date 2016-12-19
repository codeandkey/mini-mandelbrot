CC = gcc
CFLAGS = -std=c99 -Wall
LDFLAGS = -lglfw -lGL -ldl -lm -lpthread -lgmp -lmpfr

OUTPUT = mandelbrot

SOURCES = $(wildcard *.c)
OBJECTS = $(SOURCES:.c=.o)

all: $(OUTPUT)

$(OUTPUT): $(OBJECTS)
	$(CC) $(OBJECTS) $(LDFLAGS) -o $(OUTPUT)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(OUTPUT)
