# Relatório da versão otimizada da combinação de registros

## 1. Versão Inicial Sequencial
A versão inicial do código realizava a combinação de registros de dois conjuntos de dados de forma sequencial. As etapas incluíam:
- Carregamento de registros de arquivos.
- Comparações para encontrar mínimos e máximos.
- Geração de combinações de IDs.
- Cálculo de produtos e escrita em um arquivo de saída.

**Problemas Identificados na Versão Inicial:**
- Processamento sequencial com desempenho limitado em grandes conjuntos de dados (30.000 registros).
- Uso ineficiente de recursos de hardware, levando a altos tempos de execução.
- Busca linear de registros, o que aumentava a complexidade temporal.

## 2. Melhorias Aplicadas para a Versão Paralela

### 2.1. Implementação de OpenMP para Paralelismo
O código foi adaptado para usar OpenMP, permitindo o processamento paralelo em várias partes críticas:

- **Carregamento e filtração de registros**:
  - Utilização de `#pragma omp parallel for` para filtrar os registros de forma paralela.

- **Verificação de mínimos em comparações**:
  - Implementação de `#pragma omp parallel for` com `schedule(dynamic, 128)` para balancear a carga de trabalho.

- **Combinação de registros e cálculo de produtos**:
  - Uso de `#pragma omp parallel for collapse(2)` para melhorar o desempenho na combinação de registros.

### 2.2. Otimizações de Memória
- **Redução de cópias desnecessárias**:
  - Substituição de `strcpy` por `strncpy` para manipulação de strings de forma mais eficiente e segura.

- **Melhor uso de cache**:
  - Acesso a registros reorganizado para melhor localidade de cache, reduzindo o tempo de acesso à memória.

### 2.3. Filtragem de Registros Grandes e Pequenos
- **Filtragem antecipada de registros**:
  - Registros de `A` com valores menores que `THRESHOLD_CA_MIN` e registros de `B` com valores maiores que `THRESHOLD_CB_MAX` são eliminados antes das etapas de verificação mais complexas. Isso reduz a quantidade de dados processados nas etapas posteriores e melhora a eficiência.

### 2.4. Verificação Otimizada 2 a 2
- **Checagem 2 a 2 aprimorada**:
  - verificação para encontrar registros elegíveis à mínimos quando comparados 2 a 2 com otimização para parar assim que um registro se verificasse com e essa qualidade. Além disso, a lógica foi estruturada para evitar a dupla verificação do mesmo par, eliminando comparações redundantes e economizando ciclos de CPU.

### 2.5. Uso de Estruturas de Dados Paralelas
- **Tabelas hash paralelas**:
  - Cada thread cria uma tabela hash local para armazenar os registros durante a execução paralela. As tabelas são mescladas em uma etapa crítica, reduzindo a contenção de recursos.

### 2.6. Vetorização
- **Adicionada diretriz de vetorização**:
  - Uso de `#pragma omp simd` em loops relevantes para sugerir ao compilador a vetorização das operações, aumentando a eficiência.

## 3. Comparativo de Desempenho
### Análise Antes e Depois
- **Tempo de execução**:
  - A implementação paralela reduziu significativamente o tempo de execução em relação à versão sequencial.
- **Uso de CPU**:
  - O uso de múltiplos núcleos melhorou a eficiência geral, aproveitando o paralelismo de dados.
- **Complexidade de busca**:
  - A utilização de tabelas hash melhorou a busca de registros de `O(n)` para `O(1)`, acelerando as etapas de cálculo do produto.
- **Filtragem antecipada e checagem otimizada**:
  - A redução da quantidade de registros processados e a otimização na checagem 2 a 2 contribuíram para ganhos adicionais de desempenho.

## 4. Conclusão
As melhorias aplicadas, incluindo paralelismo com OpenMP, otimizações de memória, filtragem antecipada de registros e uso de tabelas hash paralelas, resultaram em um código mais eficiente e com melhor desempenho em grandes volumes de dados. Essas modificações garantiram o uso mais eficiente dos recursos de hardware, reduzindo significativamente o tempo de execução.