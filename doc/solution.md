"""
# Relatório da Versão Otimizada da Combinação de Registros

## 1. Introdução

Este relatório descreve as otimizações realizadas no código responsável pela combinação de registros de dois conjuntos de dados, visando melhorar seu desempenho e eficiência, especialmente em grandes volumes de dados. A base para este relatório é o código selecionado como ótimo, fornecido anteriormente, ao qual foram aplicadas melhorias adicionais para aumentar ainda mais a eficiência.

## 2. Versão Inicial Sequencial

A versão inicial do código realizava a combinação de registros de dois conjuntos de dados (`A` e `B`) de forma sequencial. As etapas incluíam:

- **Carregamento de registros de arquivos**: Leitura dos valores e IDs dos registros a partir de arquivos de entrada.
- **Comparações para encontrar mínimos e máximos**: Identificação dos registros com os menores e maiores valores em cada conjunto.
- **Geração de combinações de IDs**: Criação de novos IDs combinando partes dos IDs originais dos registros.
- **Cálculo de produtos e escrita em um arquivo de saída**: Cálculo de um valor `f` baseado nos registros combinados e armazenamento dos resultados.

**Problemas Identificados na Versão Inicial:**

- **Processamento sequencial**: Limitado em termos de desempenho quando aplicado a grandes conjuntos de dados (por exemplo, 30.000 registros), resultando em tempos de execução elevados.
- **Uso ineficiente de recursos de hardware**: Não aproveitava os múltiplos núcleos e capacidades de paralelismo dos processadores modernos.
- **Destruição de dados originais**: O mecanismo de filtragem alterava os valores dos registros, destruindo informações previamente escritas.
- **Busca linear de registros**: Operações de busca com complexidade temporal de `O(n)`, impactando negativamente o desempenho.

## 3. Melhorias Aplicadas para a Versão Otimizada

Com base no código escolhido como ótimo, várias otimizações foram implementadas para superar os problemas identificados.

### 3.1. Preservação dos Dados Originais

- **Introdução do campo `valid` na estrutura `Record`**:
  - Em vez de marcar registros inválidos alterando seus valores, foi adicionado um campo `valid` que indica se um registro é válido (`1`) ou inválido (`0`).
  - Isso preserva os dados originais, evitando a destruição de informações.

### 3.2. Implementação de OpenMP para Paralelismo

- **Filtragem de registros em paralelo**:
  - Uso de `#pragma omp parallel for` para filtrar os registros de `A` e `B` de forma paralela, aumentando a velocidade desta etapa.

- **Construção paralela da tabela hash**:
  - Os registros válidos de `A` e `B` são adicionados à tabela hash em paralelo, utilizando tabelas hash locais em cada thread para reduzir a contenção.

### 3.3. Otimizações de Memória

- **Redução de cópias desnecessárias**:
  - Uso de `strncpy` para manipulação de strings de forma mais eficiente e segura.

- **Melhoria da localidade de cache**:
  - Reorganização do acesso aos dados para aproveitar melhor a hierarquia de memória, reduzindo o tempo de acesso.

### 3.4. Filtragem Antecipada de Registros

- **Eliminação de registros irrelevantes**:
  - Registros de `A` com valores menores ou iguais a `THRESHOLD_CA_MIN` e registros de `B` com valores maiores ou iguais a `THRESHOLD_CB_MAX` são marcados como inválidos antes das etapas de combinação.

- **Remoção dos extremos**:
  - Após a filtragem inicial, o maior valor restante em `A` e o menor valor restante em `B` são identificados e marcados como inválidos, pois não serão usados nas combinações.

### 3.5. Uso de Estruturas de Dados Otimizadas

#### 3.5.1. Tabela Hash para Busca Eficiente

- **Construção da tabela hash com registros válidos**:
  - Os registros válidos de `A` e `B` são inseridos em uma tabela hash para permitir buscas eficientes pelo ID.

- **Busca otimizada de registros**:
  - Durante a combinação dos registros de `A` e `B`, a tabela hash é utilizada para encontrar rapidamente registros correspondentes ao ID combinado.

### 3.6. Otimizações no Loop Final

#### 3.6.1. Paralelização do Loop Final

- **O que foi feito**:
  - O loop final que combina os registros de `A` e `B` e calcula o valor `f` foi paralelizado usando OpenMP.
  - Utilizamos a diretiva `#pragma omp parallel for collapse(2) schedule(dynamic)` para paralelizar os loops aninhados que percorrem os registros de `A` e `B`.
  
- **Detalhes da implementação**:
  - A cláusula `collapse(2)` permite que os dois loops aninhados sejam tratados como um único loop de iterações, distribuindo melhor a carga de trabalho entre os threads.
  - O `schedule(dynamic)` distribui as iterações de forma dinâmica, o que é útil quando as iterações têm tempos de execução variáveis.

- **Por que foi feito**:
  - Isso permite que múltiplos threads processem diferentes combinações de registros simultaneamente, aproveitando ao máximo os recursos de hardware disponíveis e reduzindo o tempo total de processamento.

#### 3.6.2. Implementação de Buffers Locais por Thread

- **O que foi feito**:
  - Implementamos buffers de saída locais para cada thread, onde os resultados são armazenados antes de serem escritos no arquivo de saída.

- **Detalhes da implementação**:
  - Cada thread aloca um `OutputBuffer`, que é uma estrutura contendo um ponteiro para os dados, o tamanho atual e a capacidade do buffer.
  - Durante o processamento, cada thread escreve seus resultados no seu buffer local, expandindo-o conforme necessário.
  - Após o término do processamento, os buffers de todos os threads são concatenados e escritos no arquivo de saída de uma só vez.

- **Por que foi feito**:
  - Escrever diretamente no arquivo de saída dentro de múltiplos threads pode causar contenção e degradação de desempenho.
  - O uso de buffers locais elimina a necessidade de sincronização durante a escrita, pois cada thread trabalha com seu próprio buffer.
  - Isso melhora significativamente a eficiência das operações de E/S e o desempenho geral do programa.

#### 3.6.3. Correção de Comportamento Não Determinístico na Função `combine_ids`

- **Problema Identificado**:
  - O uso de uma variável estática na função `combine_ids` causava comportamento não determinístico quando múltiplos threads chamavam a função simultaneamente.
  - A função original utilizava um buffer estático para armazenar o resultado, o que não é seguro em um contexto multithread.

- **Detalhes da implementação**:
  - **Função original**:
    ```c
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
    ```
  - Múltiplos threads escreviam e liam do mesmo buffer estático `result`, levando a condições de corrida e resultados inconsistentes.

- **Solução Implementada**:
  - Modificamos a função `combine_ids` para aceitar um buffer externo onde o resultado é armazenado.
  - **Função corrigida**:
    ```c
    void combine_ids(const char *id1, const char *id2, char *result) {
        result[0] = id1[0];
        result[1] = id2[1];
        result[2] = id1[2];
        result[3] = id2[3];
        result[4] = id1[4];
        result[5] = '\0';
    }
    ```
  - Cada thread agora fornece seu próprio buffer local, eliminando a condição de corrida.

- **Ajuste nas Chamadas da Função**:
  - No loop paralelo, ajustamos a chamada para:
    ```c
    char combined_id[6];
    combine_ids(idsA[i], idsB[j], combined_id);
    ```
  - Isso garante que cada iteração do loop utilize seu próprio buffer `combined_id`, seguro para uso em paralelo.

- **Por que foi feito**:
  - Para corrigir o comportamento não determinístico causado por condições de corrida na função `combine_ids`.
  - Garantir que a função seja thread-safe, permitindo resultados consistentes entre as execuções.

### 3.7. Ordenação do Arquivo de Saída

- **O que foi feito**:
  - Ordenamos o arquivo de saída com base na coluna `f` em ordem crescente, conforme especificado no projeto.

- **Detalhes da implementação**:
  - O comando utilizado é:
    ```c
    system("(head -n 1 ./output/par.csv && tail -n +2 ./output/par.csv | sort -t, -k6 -n) > ./output/sorted_par.csv");
    ```
  - Isso preserva o cabeçalho do arquivo (`head -n 1`) e ordena o restante das linhas (`tail -n +2`) com base na sexta coluna (`-k6`), que corresponde ao valor `f`, de forma numérica (`-n`).

- **Por que foi feito**:
  - Garantir que o arquivo de saída esteja ordenado corretamente é crucial para atender aos requisitos do projeto.
  - A ordenação com base na coluna `f` permite que os resultados sejam analisados e interpretados corretamente.

## 4. Considerações sobre o Código Otimizado

- **Desempenho Aprimorado**:
  - As otimizações implementadas, especialmente a paralelização do loop final e a utilização de buffers locais, resultaram em um código significativamente mais rápido.
  - O tempo de processamento foi reduzido, permitindo lidar com conjuntos de dados maiores de forma eficiente.

- **Paralelismo Efetivo**:
  - O código agora aproveita plenamente os múltiplos núcleos disponíveis, distribuindo a carga de trabalho de maneira equilibrada entre os threads.
  - A remoção de seções críticas desnecessárias evitou gargalos que limitavam o desempenho.

- **Segurança e Confiabilidade**:
  - A correção do comportamento não determinístico na função `combine_ids` assegura que o programa produza resultados consistentes e corretos em todas as execuções.

- **Eficiência nas Operações de E/S**:
  - A implementação de buffers locais por thread reduziu a contenção durante a escrita no arquivo de saída, melhorando a eficiência das operações de entrada e saída.

- **Preservação dos Dados**:
  - O uso do campo `valid` assegura que os dados originais não sejam alterados, mantendo a integridade das informações e permitindo possíveis reutilizações dos dados sem a necessidade de recarregá-los.

## 5. Conclusão

As melhorias implementadas transformaram o código original em uma versão altamente otimizada, capaz de processar grandes volumes de dados de forma eficiente. A combinação de paralelismo, otimizações de memória, correções de problemas de thread safety e uso de estruturas de dados adequadas resultou em:

- **Desempenho Superior**: Processamento mais rápido e eficiente, com melhor utilização dos recursos de hardware.
- **Escalabilidade**: O código agora é capaz de lidar com conjuntos de dados maiores sem comprometer o desempenho, graças às otimizações aplicadas.
- **Confiabilidade e Consistência**: A correção de problemas não determinísticos garante resultados confiáveis em todas as execuções.