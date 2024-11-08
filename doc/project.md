# Projeto

Em muitas aplicações paralelas temos o seguinte padrão de processamento de dados:

1 - ler informações em mais de uma estrutura de dados;
2 - realizar o processamento de tais dados;
3 - armazenar a informação de saída.

Neste projeto, iremos simular uma aplicação real via um processamento dummy. O objetivo é evitar gastar tempo em compreender a aplicação e focar na otimização do runtime, objetivo do projeto.

Todos os tipos de otimizações são permitidas, inclusive decompor a aplicação em serviços isolados, escolher estruturas de dados alternativas, dentre outras.
Também é permitido o uso do Chatgpt para auxiliar na construção da solução final, bem como para projetar a solução e obter sugestões para melhorias de desempenho.
Peço para evitar ao máximo utilizar outras linguagens, para mantermos o curso padronizado na linguagem C. Contudo, se acreditar que sua solução irá funcionar melhor com algum recurso disponível em outra linguagem, será permitido seu uso.

## Avaliação

A nota final irá considerar:

- a complexidade das abordagens testadas
- a quantidade de abordagens testadas
- runtime da solução final

O aluno receberá zero caso apresente:
- Runtime pior do que a versão sequencial não otimizada;
- Código errado, e.g., não funcional, que apresente resultados pré-computados ao invés de construir uma solução, etc.

## Base de dados

O projeto possui uma pequena base de dados:

- composta por 30 mil registros
- cada registro possui dois valores aleatórios de 0 à 1;
- cada registro possui um ID alfabético de exatamente 5 caracteres;

Essa base deve ser construída usando seed = 0, como apresentado no arquivo `db.c`.

A aplicação deve consumir os arquivos salvos em HD. É permitido mudar o formato dos arquivos, o que é considerado uma otimização que pode gerar speedup, mas não é permitido realizar qualquer pré-processamento de responsabilidade da aplicação na base de dados.

## Aplicação

Definição do operador sobre IDs:

- Considere dois registros da base de dados, com IDs ${ID}_1 = c^1_1 c^1_2 c^1_3 c^1_4 c^1_5$ e ${ID}_2 = c^2_1 c^2_2 c^2_3 c^2_4 c^2_5$, onde cada $c_i$ é um dos 5 caracteres.
- Considere o perador binário $\circ$ tal que ${ID}_1 \circ {ID}_2 = c^1_1 c^2_2 c^1_3 c^2_4 c^1_5$.

Definições de composições de registros da base de dados:

- Considere $Ca$ todas as possíveis 30000^2 - 30000 combinações de números aleatórios do arquivo A.
- Considere $Cb$ todas as possíveis 30000^2 - 30000 combinações de números aleatórios do arquivo B.
- Considere $Cab$ o produto cartesiano dos registros de $Ca$ com $Cb$, que resultam em ~ 8 * 10^17 novos registros.

Assim, cada registro de $Ca$ terá dois IDs: ${ID}_{a1}$ e ${ID}_{a2}$ e dois valores aleatórios ${a1}$ e ${a2}$.
Por simetria, $Cb$ terá ${ID}_{b1}$ e ${ID}_{b2}$ e dois valores aleatórios ${b1}$ e ${b2}$.

Já $Cab$ terá 4 IDs e 4 valores aleatórios: ${ID}_{a1}$, ${ID}_{a2}$, ${ID}_{b1}$ e ${ID}_{b2}$; e ${a1}$, ${a2}$, ${b1}$ e ${b2}$.

### Descrição

As seguintes computações devem ser realizadas:

- Calcule o mínimo $a_m$ dos valores aleatórios de cada registro de $Ca$ e ignore este registro caso $a_m \leq 0.25$.
- Calcule o máximo $b_M$ dos valores aleatórios de cada registro de $Cb$ e ignore este registro caso $b_M \geq 0.75$.
- Para cada registro de $Cab$, identifique o primeiro registro na base de dados cujo ID tenha o valor ${ID}' = {ID}_{a_m} \circ {ID}_{b_M}$ e retorne o produto $p$ dos valores aleatórios relacionados aos arquivos A e B. Caso este registro não exista, ignore a computação desse registro de $Cab$.
- Para cada registro de $Cab$ resultado, calcule o produto final $f = a_m * b_M * p$.

A computação deverá retornar um arquivo no formto .csv, separados por vírgula, com os registros $Cab$ resultantes, contendo as seguintes colunas: ID_a_m, ID_b_M, ID', a_m, b_M, f.

O arquivo deverá ter os registros ordenados de forma ascendentes com base nos valores da coluna f.

O arquivo `seq.c` apresenta uma implementação trivial da aplicação de forma sequencial.

## Runtime

Para a captura do runtime, considere desde o tempo de leitura da estrutura de dados até o final da escrita do arquivo resultante, bem como as desalocações de memória.

## Relatório

O código final deve ser postado de forma organizada no github.

Faça no mínimo a seguinte organização de arquivos e diretórios:

- db: pasta com a base de dados
- src: pasta com o código fonte da sua solução
- doc: documentação sobre a solução
- tutorial: exemplo com comandos de terminal passo-a-passo de como instalar o compilador, bibliotecas e outros necessário, bem como utilizar e executar sua aplicação
- README.md: arquivo descrevendo a organização do repositório e indicando os links para as partes principais do seu projeto

Dentro da pasta `doc`, descreva sua solução em Markdown. Pode-se utilizar imagens svg dentro do Markdown.

O aluno deverá enviar o link para o repositório de sua solução no github.
