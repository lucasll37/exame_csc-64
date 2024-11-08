#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "uthash.h" // Biblioteca para a implementação de hashtable

#define NUM_RECORDS 30000
#define THRESHOLD_CA_MIN 0.25f
#define THRESHOLD_CB_MAX 0.75f

typedef struct {
    char id[6];
    float value;
    UT_hash_handle hh; // Handle para a tabela hash
} HashRecord;

typedef struct {
    char id[6];
    float value;
} Record;

double start_time, end_time;

void load_records(const char *filename, Record *records) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Erro ao abrir o arquivo %s.\n", filename);
        exit(1);
    }
    for (int i = 0; i < NUM_RECORDS; ++i) {
        fscanf(file, "%f", &records[i].value);
    }
    fclose(file);
}

void load_ids(const char *filename, char (*ids)[6]) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Erro ao abrir o arquivo %s.\n", filename);
        exit(1);
    }
    for (int i = 0; i < NUM_RECORDS; ++i) {
        fscanf(file, "%s", ids[i]);
    }
    fclose(file);
}

char *combine_ids(const char *id1, const char *id2) {
    static char result[6];
    result[0] = id1[0];
    result[1] = id2[1];
    result[2] = id1[2];
    result[3] = id2[3];
    result[4] = id1[4];
    result[5] = '\0';
    return result;
}

int main() {
    start_time = omp_get_wtime();
    printf("\n\nNumber of cores available: %d\n", omp_get_max_threads());

    // Aloca memória dinamicamente para os registros e IDs
    Record *recordsA = (Record *)malloc(NUM_RECORDS * sizeof(Record));
    Record *recordsB = (Record *)malloc(NUM_RECORDS * sizeof(Record));
    char(*idsA)[6] = (char(*)[6])malloc(NUM_RECORDS * sizeof(*idsA));
    char(*idsB)[6] = (char(*)[6])malloc(NUM_RECORDS * sizeof(*idsB));

    if (!recordsA || !recordsB || !idsA || !idsB) {
        fprintf(stderr, "Erro ao alocar memória.\n");
        exit(1);
    }

    load_records("./db/A.txt", recordsA);
    load_records("./db/B.txt", recordsB);
    load_ids("./db/ids.txt", idsA);
    load_ids("./db/ids.txt", idsB);

    // Filtra registros de A e B em paralelo
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < NUM_RECORDS; ++i) {
        if (recordsA[i].value <= THRESHOLD_CA_MIN) {
            recordsA[i].value = -1.0f; // Marca registros que não passam no filtro
        }
        if (recordsB[i].value >= THRESHOLD_CB_MAX) {
            recordsB[i].value = -1.0f; // Marca registros que não passam no filtro
        }
    }

    // Cria uma tabela hash para armazenar os IDs e valores
    HashRecord *hash_table = NULL;

    // Adiciona registros de A à tabela hash
    #pragma omp parallel
    {
        HashRecord *local_table = NULL;

        #pragma omp for nowait
        for (int i = 0; i < NUM_RECORDS; ++i) {
            if (recordsA[i].value != -1.0f) {
                HashRecord *entry = (HashRecord *)malloc(sizeof(HashRecord));
                strncpy(entry->id, idsA[i], 5);
                entry->id[5] = '\0';
                entry->value = recordsA[i].value;
                HASH_ADD_STR(local_table, id, entry);
            }
        }

        #pragma omp critical
        {
            HashRecord *current_entry, *tmp;
            HASH_ITER(hh, local_table, current_entry, tmp) {
                HASH_ADD_STR(hash_table, id, current_entry);
            }
        }
    }

    // Adiciona registros de B à tabela hash
    #pragma omp parallel
    {
        HashRecord *local_table = NULL;

        #pragma omp for nowait
        for (int i = 0; i < NUM_RECORDS; ++i) {
            if (recordsB[i].value != -1.0f) {
                HashRecord *entry = (HashRecord *)malloc(sizeof(HashRecord));
                strncpy(entry->id, idsB[i], 5);
                entry->id[5] = '\0';
                entry->value = recordsB[i].value;
                HASH_ADD_STR(local_table, id, entry);
            }
        }

        #pragma omp critical
        {
            HashRecord *current_entry, *tmp;
            HASH_ITER(hh, local_table, current_entry, tmp) {
                HASH_ADD_STR(hash_table, id, current_entry);
            }
        }
    }

    // Etapa final: combinação e cálculo do produto
    FILE *output = fopen("./output/output_par.csv", "w");

    if (!output) {
        fprintf(stderr, "Erro ao criar o arquivo de saída.\n");
        free(recordsA);
        free(recordsB);
        free(idsA);
        free(idsB);
        return 1;
    }

    fprintf(output, "ID_a_m,ID_b_M,ID',a_m,b_M,f\n");

    #pragma omp parallel for collapse(2) schedule(dynamic, 64)
    for (int i = 0; i < NUM_RECORDS; ++i) {
        for (int j = 0; j < NUM_RECORDS; ++j) {
            if (i != j && recordsA[i].value != -1.0f && recordsB[j].value != -1.0f) {
                char combined_id[6];
                strcpy(combined_id, combine_ids(idsA[i], idsB[j]));

                float product = recordsA[i].value * recordsB[j].value;

                HashRecord *found;

                #pragma omp critical
                {
                    HASH_FIND_STR(hash_table, combined_id, found);
                    if (found) {
                        float f = product * found->value;
                        fprintf(output, "%s,%s,%s,%f,%f,%f\n", idsA[i], idsB[j], combined_id, recordsA[i].value, recordsB[j].value, f);
                    }
                }
            }
        }
    }

    fclose(output);
    system("sort -t, -k6 -n ./output/output_par.csv -o ./output/sorted_output_par.csv");

    // Libera a memória
    HashRecord *current_entry, *tmp;
    HASH_ITER(hh, hash_table, current_entry, tmp) {
        HASH_DEL(hash_table, current_entry);
        free(current_entry);
    }
    free(recordsA);
    free(recordsB);
    free(idsA);
    free(idsB);

    printf("Processamento completo. Resultados salvos em sorted_output_par.csv\n");

    end_time = omp_get_wtime();
    printf("Tempo de processamento: %.2f segundos\n", end_time - start_time);

    return 0;
}
