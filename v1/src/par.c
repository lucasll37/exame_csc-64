#include <float.h>  // Para FLT_MAX
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "uthash.h"  // Biblioteca para a implementação de hashtable

#define NUM_RECORDS 30000
#define THRESHOLD_CA_MIN 0.25f
#define THRESHOLD_CB_MAX 0.75f

typedef struct {
    char id[6];
    int position;
    float value;
    UT_hash_handle hh;  // Handle para a tabela hash
} HashRecord;

typedef struct {
    char id[6];
    float value;
    int valid;  // 1 se válido, 0 se inválido
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
        records[i].valid = 1;  // Inicializa como válido
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

void combine_ids(const char *id1, const char *id2, char *result) {
    result[0] = id1[0];
    result[1] = id2[1];
    result[2] = id1[2];
    result[3] = id2[3];
    result[4] = id1[4];
    result[5] = '\0';
}

int main() {
    start_time = omp_get_wtime();
    printf("\n\nNúmero de núcleos disponíveis: %d\n", omp_get_max_threads());

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
            recordsA[i].valid = 0;  // Marca registros que não passam no filtro
        }
        if (recordsB[i].value >= THRESHOLD_CB_MAX) {
            recordsB[i].valid = 0;  // Marca registros que não passam no filtro
        }
    }

    // Encontra e remove o maior valor em A após o filtro
    float maxA = -1.0f;
    int maxAIndex = -1;

    for (int i = 0; i < NUM_RECORDS; ++i) {
        if (recordsA[i].valid && recordsA[i].value > maxA) {
            maxA = recordsA[i].value;
            maxAIndex = i;
        }
    }

    if (maxAIndex != -1) {
        recordsA[maxAIndex].valid = 0;  // Marca o maior valor como removido
    }

    // Encontra e remove o menor valor em B após o filtro
    float minB = FLT_MAX;  // Valor inicial alto para garantir a busca do mínimo
    int minBIndex = -1;

    for (int i = 0; i < NUM_RECORDS; ++i) {
        if (recordsB[i].valid && recordsB[i].value < minB) {
            minB = recordsB[i].value;
            minBIndex = i;
        }
    }
    if (minBIndex != -1) {
        recordsB[minBIndex].valid = 0;  // Marca o menor valor como removido
    }

    // Cria uma tabela hash para armazenar os IDs e valores
    HashRecord *hash_table = NULL;

// Adiciona registros de A à tabela hash
#pragma omp parallel
    {
        HashRecord *local_table = NULL;

#pragma omp for nowait
        for (int i = 0; i < NUM_RECORDS; ++i) {
            if (recordsA[i].valid) {
                HashRecord *entry = (HashRecord *)malloc(sizeof(HashRecord));
                strncpy(entry->id, idsA[i], 5);
                entry->id[5] = '\0';
                entry->position = i;
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
            if (recordsB[i].valid) {
                HashRecord *entry = (HashRecord *)malloc(sizeof(HashRecord));
                strncpy(entry->id, idsB[i], 5);
                entry->id[5] = '\0';
                entry->position = i;
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
    FILE *output = fopen("./output/par.csv", "w");

    if (!output) {
        fprintf(stderr, "Erro ao criar o arquivo de saída.\n");
        free(recordsA);
        free(recordsB);
        free(idsA);
        free(idsB);
        return 1;
    }

    fprintf(output, "ID_a_m,ID_b_M,ID',a_m,b_M,f\n");

    // Cada thread terá seu próprio buffer de saída
    typedef struct {
        char *data;
        size_t size;
        size_t capacity;
    } OutputBuffer;

    int num_threads = omp_get_max_threads();
    OutputBuffer *buffers = (OutputBuffer *)calloc(num_threads, sizeof(OutputBuffer));

#pragma omp parallel
    {
        int thread_num = omp_get_thread_num();
        OutputBuffer *buf = &buffers[thread_num];

        // Inicializa o buffer
        buf->capacity = 1024;
        buf->data = (char *)malloc(buf->capacity);
        buf->size = 0;

#pragma omp for collapse(2) schedule(dynamic)
        for (int i = 0; i < NUM_RECORDS; ++i) {
            for (int j = 0; j < NUM_RECORDS; ++j) {
                if (recordsA[i].valid && recordsB[j].valid) {
                    char combined_id[6];
                    // Uso da função atualizada
                    combine_ids(idsA[i], idsB[j], combined_id);

                    float product = recordsA[i].value * recordsB[j].value;

                    // HASH_FIND_STR é thread-safe para leitura
                    HashRecord *found;
                    HASH_FIND_STR(hash_table, combined_id, found);

                    if (found) {
                        float f = product * recordsA[found->position].value * recordsB[found->position].value;

                        // Escreve no buffer local
                        char line[256];
                        int len = snprintf(line, sizeof(line), "%s,%s,%s,%f,%f,%f\n",
                                           idsA[i], idsB[j], combined_id, recordsA[i].value, recordsB[j].value, f);

                        // Redimensiona o buffer se necessário
                        if (buf->size + len >= buf->capacity) {
                            buf->capacity *= 2;
                            buf->data = (char *)realloc(buf->data, buf->capacity);
                        }
                        memcpy(buf->data + buf->size, line, len);
                        buf->size += len;
                    }
                }
            }
        }
    }

    // Escreve os buffers no arquivo de saída
    for (int i = 0; i < num_threads; ++i) {
        fwrite(buffers[i].data, 1, buffers[i].size, output);
        free(buffers[i].data);
    }
    free(buffers);
    fclose(output);

    // Ordena o arquivo com base na coluna 'f'
    system("(head -n 1 ./output/par.csv && tail -n +2 ./output/par.csv | sort -t, -k6 -n) > ./output/unique_sorted_par.csv");

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

    printf("Processamento completo (N = %d). Resultados salvos em ./output/unique_sorted_par.csv\n", NUM_RECORDS);

    end_time = omp_get_wtime();
    printf("Tempo de processamento: %.2f segundos\n", end_time - start_time);

    return 0;
}
