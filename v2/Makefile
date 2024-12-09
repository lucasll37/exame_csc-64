CC = gcc
MPICC = mpicc
CFLAGS = -Wall -fopenmp
INCLUDES = -I./libs/uthash/include -I/lib/x86_64-linux-gnu/openmpi/include
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

comp_par: $(BUILDDIR)/par.o
	$(MPICC) $(CFLAGS) $(INCLUDES) $^ -o $(TARGET_PARALLEL) -lstdc++

$(BUILDDIR)/par.o: $(SRCDIR)/par.c | folder_build
	$(MPICC) $(CFLAGS) $(INCLUDES) -c $< -o $@

comp_seq: $(SRCDIR)/seq.c | folder_build
	$(CC) $(CFLAGS) $(INCLUDES) $< -o $(TARGET_SEQ)

comp_db: $(SRCDIR)/db.c | folder_build
	$(CC) $(CFLAGS) $(INCLUDES) $< -o $(TARGET_DB)

db: folder_db | comp_db
	$(TARGET_DB)

clean:
	rm -rf $(BUILDDIR) $(OUTPUTDIR) $(DBDIR)