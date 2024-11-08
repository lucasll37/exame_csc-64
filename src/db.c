#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NUM_RECORDS 100
#define ID_LENGTH 5

// Função para gerar uma string alfabética maiúscula aleatória de tamanho fixo
void generate_random_id(char *id) {
  static const char alphabet[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
  for (int i = 0; i < ID_LENGTH; ++i) {
    id[i] = alphabet[rand() % (sizeof(alphabet) - 1)];
  }
  id[ID_LENGTH] = '\0';
}

int main() {
  // Define o seed para garantir a reproducibilidade
  srand(0);

  FILE *fileA = fopen("./db/A.txt", "w");
  FILE *fileB = fopen("./db/B.txt", "w");
  FILE *id_file = fopen("./db/ids.txt", "w");

  if (!fileA || !fileB || !id_file) {
    fprintf(stderr, "Erro ao abrir os arquivos.\n");
    return 1;
  }

  for (int i = 0; i < NUM_RECORDS; ++i) {
    // Gera um número aleatório entre 0 e 1
    double random_num_A = (double)rand() / RAND_MAX;
    double random_num_B = (double)rand() / RAND_MAX;
    fprintf(fileA, "%f\n", random_num_A);
    fprintf(fileB, "%f\n", random_num_B);

    // Gera uma string aleatória alfabética maiúscula de 5 caracteres
    char id[ID_LENGTH + 1];
    generate_random_id(id);
    fprintf(id_file, "%s\n", id);
  }

  fclose(fileA);
  fclose(fileB);
  fclose(id_file);

  printf("Base de dados gerada com sucesso!\n");

  return 0;
}
