MAKEFLAGS+=-r -j

UNAME=$(shell uname)

CFLAGS=-g -Wall -Wpointer-arith -Werror # -O3 #-ffast-math
LDFLAGS=-lm

BUILD=build
SOURCES=$(wildcard src/*.c)

OBJS=$(SOURCES:%=$(BUILD)/%.o)



ifeq ($(UNAME),Darwin)
	FRAMEWORKS_DIR=/Library/Developer/CommandLineTools/SDKs/MacOSX15.2.sdk/System/Library/Frameworks/
	ACCELERATE_HEADERS=$(FRAMEWORKS_DIR)/Accelerate.framework/Versions/A/Headers
	CFLAGS+=-I$(ACCELERATE_HEADERS) -DACCELERATE_NEW_LAPACK
	LDFLAGS+=-dynamiclib -framework Accelerate
else
	LDFLAGS+=-shared
endif

ifeq ($(UNAME),Darwin)
	LIBRARY_NAME=libeel.dylib
else
	LIBRARY_NAME=libeel.so
endif
LIBRARY=$(BUILD)/$(LIBRARY_NAME)

all: $(LIBRARY)

format:
	clang-format -i src/*


$(LIBRARY): $(OBJS)
	$(CC) $^ $(LDFLAGS) -o $@

$(BUILD)/%.c.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $< $(CFLAGS) -c -MMD -MP -o $@


clean:
	rm -rf $(BUILD)


.PHONY: all clean format