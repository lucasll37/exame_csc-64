#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define NUM_RECORDS 30000
#define THRESHOLD_CA_MIN 0.25f
#define THRESHOLD_CB_MAX 0.75f

typedef struct {
  char id[6];
  float value;
} Record;

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

int compare_records(const void *a, const void *b) {
  float f1 = *((float *)a);
  float f2 = *((float *)b);
  if (f1 < f2)
    return -1;
  if (f1 > f2)
    return 1;
  return 0;
}

int main() {
  clock_t start_time, end_time;
  start_time = clock();

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

  FILE *output = fopen("./output/seq.csv", "w");
  if (!output) {
    fprintf(stderr, "Erro ao abrir o arquivo de saída.\n");
    free(recordsA);
    free(recordsB);
    free(idsA);
    free(idsB);
    return 1;
  }
  fprintf(output, "ID_a_m,ID_b_M,ID',a_m,b_M,f\n");

  for (int Ca1 = 0; Ca1 < NUM_RECORDS; ++Ca1) {
    for (int Ca2 = 0; Ca2 < NUM_RECORDS; ++Ca2) {
      if (Ca1 != Ca2) {
        char *id_minCa =
            recordsA[Ca1].value < recordsA[Ca2].value ? idsA[Ca1] : idsA[Ca2];
        float minCa = recordsA[Ca1].value < recordsA[Ca2].value
                          ? recordsA[Ca1].value
                          : recordsA[Ca2].value;
        if (minCa <= THRESHOLD_CA_MIN) {
          continue;
        }

        for (int Cb1 = 0; Cb1 < NUM_RECORDS; ++Cb1) {
          for (int Cb2 = 0; Cb2 < NUM_RECORDS; ++Cb2) {
            if (Cb1 != Cb2) {
              char *id_maxCb = recordsB[Cb1].value > recordsB[Cb2].value
                                   ? idsB[Cb1]
                                   : idsB[Cb2];
              float maxCb = recordsB[Cb1].value > recordsB[Cb2].value
                                ? recordsB[Cb1].value
                                : recordsB[Cb2].value;
              if (maxCb >= THRESHOLD_CB_MAX) {
                continue;
              }

              char *combined_id = combine_ids(id_minCa, id_maxCb);
              float product = minCa * maxCb;

              // Check if the combined_id exists in the records
              int found = 0;
              for (int i = 0; i < NUM_RECORDS; ++i) {
                if (strcmp(combined_id, idsA[i]) == 0 ||
                    strcmp(combined_id, idsB[i]) == 0) {
                  product *= recordsA[i].value * recordsB[i].value;
                  found = 1;
                  break;
                }
              }
              if (!found) {
                continue;
              }

              // fprintf(output, "%s,%s,%s,%f,%f,%f\n", idsA[Ca1], idsB[Cb1],
              //         combined_id, minCa, maxCb, product);
              fprintf(output, "%s,%s,%s,%f,%f,%f\n", id_minCa, id_maxCb,
                      combined_id, minCa, maxCb, product);
            }
          }
        }
      }
    }
  }

  fclose(output);

  // Ordena os registros com base no valor da coluna f
  // NOTE: This is a simplified approach; for large datasets, external sorting
  // would be more appropriate
    system("(head -n 1 ./output/seq.csv && tail -n +2 ./output/seq.csv | sort -t, -k6 -n) > ./output/sorted_seq.csv");
    system("awk '!seen[$0]++' ./output/sorted_seq.csv > ./output/unique_sorted_seq.csv");
  // Descrição do comando:
  // sort: O comando para ordenar.
  // -t,: Define a vírgula (,) como delimitador de campo.
  // -k6: Especifica que a ordenação deve ser feita com base no sexto campo.
  // -n: Realiza uma ordenação numérica.
  // output.csv: O arquivo de entrada.
  // -o sorted_output.csv: Especifica o arquivo de saída.

  // Libera a memória alocada dinamicamente
  free(recordsA);
  free(recordsB);
  free(idsA);
  free(idsB);

  printf("Processamento completo (N = %d). Resultados salvos em ./output/seq.csv\n", NUM_RECORDS);
  end_time = clock(); // Fim da medição de tempo
  double time_spent = (double)(end_time - start_time) / CLOCKS_PER_SEC;
  printf("Processing time: %.2f seconds\n", time_spent);
  return 0;
}