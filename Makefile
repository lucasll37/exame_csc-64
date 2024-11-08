CC = gcc
CFLAGS = -Wall -fopenmp
INCLUDES = -I./libs/uthash/include
SRCDIR = src
BUILDDIR = build
DBDIR = db
OUTPUTDIR = output
TARGET_PARALLEL = $(BUILDDIR)/par
TARGET_SEQ = $(BUILDDIR)/seq
TARGET_DB = $(BUILDDIR)/db

all: folder_build folder_output db comp_seq comp_par
	@echo "All was compiled"

folder_build:
	@mkdir -p $(BUILDDIR)

folder_db:
	@mkdir -p $(DBDIR)

folder_output:
	@mkdir -p $(OUTPUTDIR)

comp_par: $(SRCDIR)/par.c | folder_build
	$(CC) $(CFLAGS) $(INCLUDES) $< -o $(TARGET_PARALLEL)

comp_seq: $(SRCDIR)/seq.c | folder_build
	$(CC) $(CFLAGS) $(INCLUDES) $< -o $(TARGET_SEQ)

comp_db: $(SRCDIR)/db.c | folder_build
	$(CC) $(CFLAGS) $(INCLUDES) $< -o $(TARGET_DB)

db: folder_db | comp_db
	$(TARGET_DB)

seq: folder_output | comp_seq
	$(TARGET_SEQ)

par: folder_output | comp_par
	$(TARGET_PARALLEL)

clean:
	rm -rf $(BUILDDIR) $(DBDIR) $(OUTPUTDIR)

.PHONY: clean
