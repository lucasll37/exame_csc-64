#include <float.h>
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "uthash.h"

#define NUM_RECORDS 30000
#define THRESHOLD_CA_MIN 0.25f
#define THRESHOLD_CB_MAX 0.75f

typedef struct
{
    char id[6];
    int position;
    float value;
    UT_hash_handle hh;
} HashRecord;

typedef struct
{
    char id[6];
    float value;
    int valid;
} Record;

typedef struct
{
    char *data;
    size_t size;
    size_t capacity;
} OutputBuffer;

double start_time, end_time;

void load_records(const char *filename, Record *records, int start_index, int end_index) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Erro ao abrir o arquivo %s.\n", filename);
        exit(1);
    }
    for (int i = 0; i < start_index; ++i) {
        fscanf(file, "%*f");  // Skip records until start_index
    }
    for (int i = start_index; i < end_index; ++i) {
        fscanf(file, "%f", &records[i - start_index].value);
        records[i - start_index].valid = 1;
    }
    fclose(file);
}

void load_ids(const char *filename, char (*ids)[6], int start_index, int end_index) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Erro ao abrir o arquivo %s.\n", filename);
        exit(1);
    }
    for (int i = 0; i < start_index; ++i) {
        fscanf(file, "%*s");  // Skip ids until start_index
    }
    for (int i = start_index; i < end_index; ++i) {
        fscanf(file, "%s", ids[i - start_index]);
    }
    fclose(file);
}

void combine_ids(const char *id1, const char *id2, char *result) {
    result[0] = id1[0];
    result[1] = id2[1];
    result[2] = id1[2];
    result[3] = id2[3];
    result[4] = id1[4];
    result[5] = '\0';
}

Record *create_record(Record *record, int start_index, int end_index) {
    Record *new_record = (Record *)malloc((end_index - start_index) * sizeof(Record));
    for (int i = start_index; i < end_index; ++i) {
        new_record[i - start_index].value = record[i].value;
        new_record[i - start_index].valid = record[i].valid;
    }
    return new_record;
}

void dummy_processing(Record *recordsA, Record *recordsB, Record *global_recordsA, Record *global_recordsB, char (*idsA)[6], char (*idsB)[6], char (*global_idsB)[6], int num_records, int rank, HashRecord *global_hash_table) {
    char filename[256];
    snprintf(filename, sizeof(filename), "./output/par_%d.csv", rank);
    FILE *output = fopen(filename, "w");
    if (!output) {
        fprintf(stderr, "Erro ao criar o arquivo de saída %s.\n", filename);
        exit(1);
    }
    fprintf(output, "ID_a_m,ID_b_M,ID',a_m,b_M,f\n");

#pragma omp parallel
    {
        char line[256];

#pragma omp for collapse(2) schedule(dynamic)
        for (int i = 0; i < num_records; ++i) {
            for (int j = 0; j < NUM_RECORDS; ++j) {
                if (recordsA[i].valid && global_recordsB[j].valid) {
                    char combined_id[6];

                    combine_ids(idsA[i], global_idsB[j], combined_id);

                    float p = recordsA[i].value * global_recordsB[j].value;

                    HashRecord *found;

#pragma omp critical
                    {
                        HASH_FIND_STR(global_hash_table, combined_id, found);

                        if (found) {
                            float f = p * global_recordsA[found->position].value * global_recordsB[found->position].value;
                            int len = snprintf(line, sizeof(line), "%s,%s,%s,%f,%f,%f\n",
                                               idsA[i], global_idsB[j], combined_id, recordsA[i].value, global_recordsB[j].value, f);
                            fwrite(line, 1, len, output);
                        }
                    }
                }
            }
        }
    }

    fclose(output);
}

void create_global_hash_table(Record *recordsA, Record *recordsB, char (*idsA)[6], char (*idsB)[6], int num_records, HashRecord **global_hash_table) {
    for (int i = 0; i < num_records; ++i) {
        if (recordsA[i].valid) {
            HashRecord *entry = (HashRecord *)malloc(sizeof(HashRecord));
            strncpy(entry->id, idsA[i], 5);
            entry->id[5] = '\0';
            entry->position = i;
            entry->value = recordsA[i].value;
            HASH_ADD_STR(*global_hash_table, id, entry);
        }
        if (recordsB[i].valid) {
            HashRecord *entry = (HashRecord *)malloc(sizeof(HashRecord));
            strncpy(entry->id, idsB[i], 5);
            entry->id[5] = '\0';
            entry->position = i;
            entry->value = recordsB[i].value;
            HASH_ADD_STR(*global_hash_table, id, entry);
        }
    }
}

void free_global_hash_table(HashRecord *global_hash_table) {
    HashRecord *current_entry, *tmp;
    HASH_ITER(hh, global_hash_table, current_entry, tmp) {
        HASH_DEL(global_hash_table, current_entry);
        free(current_entry);
    }
}

int main(int argc, char *argv[]) {
    start_time = omp_get_wtime();
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    HashRecord *global_hash_table = NULL;
    char *buffer = NULL;

    int records_per_proc = NUM_RECORDS / size;
    int remainder = NUM_RECORDS % size;
    int start_index = rank * records_per_proc + (rank < remainder ? rank : remainder);
    int end_index = start_index + records_per_proc + (rank < remainder ? 1 : 0);

    int local_num_records = end_index - start_index;

    Record *global_recordsA = (Record *)malloc(NUM_RECORDS * sizeof(Record));
    Record *global_recordsB = (Record *)malloc(NUM_RECORDS * sizeof(Record));
    char(*global_idsA)[6] = (char(*)[6])malloc(NUM_RECORDS * sizeof(*global_idsA));
    char(*global_idsB)[6] = (char(*)[6])malloc(NUM_RECORDS * sizeof(*global_idsB));

    char(*idsA)[6] = (char(*)[6])malloc(local_num_records * sizeof(*idsA));
    char(*idsB)[6] = (char(*)[6])malloc(local_num_records * sizeof(*idsB));

    if (!idsA || !idsB) {
        fprintf(stderr, "Erro ao alocar memória.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    load_records("./db/A.txt", global_recordsA, 0, NUM_RECORDS);
    load_records("./db/B.txt", global_recordsB, 0, NUM_RECORDS);
    load_ids("./db/ids.txt", idsA, start_index, end_index);
    load_ids("./db/ids.txt", global_idsA, 0, NUM_RECORDS);
    load_ids("./db/ids.txt", idsB, start_index, end_index);
    load_ids("./db/ids.txt", global_idsB, 0, NUM_RECORDS);

    create_global_hash_table(global_recordsA, global_recordsB, global_idsA, global_idsB, NUM_RECORDS, &global_hash_table);

#pragma omp parallel for schedule(static)
    for (int i = 0; i < NUM_RECORDS; ++i) {
        if (global_recordsA[i].value <= THRESHOLD_CA_MIN) {
            global_recordsA[i].valid = 0;
        }
        if (global_recordsB[i].value >= THRESHOLD_CB_MAX) {
            global_recordsB[i].valid = 0;
        }
    }

    float maxA = -1.0f;
    int maxAIndex = -1;

    for (int i = 0; i < NUM_RECORDS; ++i) {
        if (global_recordsA[i].valid && global_recordsA[i].value > maxA) {
            maxA = global_recordsA[i].value;
            maxAIndex = i;
        }
    }

    if (maxAIndex != -1) {
        global_recordsA[maxAIndex].valid = 0;
    }

    float minB = FLT_MAX;
    int minBIndex = -1;

    for (int i = 0; i < NUM_RECORDS; ++i) {
        if (global_recordsB[i].valid && global_recordsB[i].value < minB) {
            minB = global_recordsB[i].value;
            minBIndex = i;
        }
    }
    if (minBIndex != -1) {
        global_recordsB[minBIndex].valid = 0;
    }

    Record *recordsA = create_record(global_recordsA, start_index, end_index);
    Record *recordsB = create_record(global_recordsB, start_index, end_index);

    MPI_Barrier(MPI_COMM_WORLD);

    dummy_processing(recordsA, recordsB, global_recordsA, global_recordsB, idsA, idsB, global_idsB, local_num_records, rank, global_hash_table);

    free(recordsA);
    free(recordsB);
    free(idsA);
    free(idsB);
    free(buffer);

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        free_global_hash_table(global_hash_table);

        // Merge all individual files into a single file
        system("head -n 1 ./output/par_0.csv > ./output/par.csv");
        for (int i = 0; i < size; ++i) {
            char command[256];
            snprintf(command, sizeof(command), "tail -n +2 ./output/par_%d.csv >> ./output/par.csv", i);
            system(command);
        }

        system("rm ./output/par_*.csv");
        system("(head -n 1 ./output/par.csv && tail -n +2 ./output/par.csv | sort -t, -k6 -n) > ./output/sorted_par.csv");
        system("awk '!seen[$0]++' ./output/sorted_par.csv > ./output/unique_sorted_par.csv");
        printf("Processamento completo (N = %d). Resultados salvos em ./output/unique_sorted_par.csv\n", NUM_RECORDS);
    }

    MPI_Finalize();

    end_time = omp_get_wtime();
    if (rank == 0) {
        printf("Tempo de processamento: %.2f segundos\n", end_time - start_time);
    }

    return 0;
}