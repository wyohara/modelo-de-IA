
# Modelo básico de IA para estudo  

Este é um projeto de um tensor para modelo de inteligência artificial basico para estudo. O modelo usa como base um tensor básico inicialmente pensado segundo o artigo "Attention is all you need" [(aqui)](/bibliografia/Attention_is_all_you_need.pdf).  

## O Tensor  

Segundo o artigo “Attention Is All You Need” foi proposto um modelo de rede neural que substitui redes recorrentes (RNNs) e convolucionais (CNNs) por um mecanismo de atenção pura, chamado **Transformer**.
As principais características são: 
- O mecanismo de **autoatenção** permite relacionar duas posições quaisquer da sequência de dados com custo constante,diferente de RNNs e CNNs que aumentavam o custo a medida que aumentavam os dados, facilitando a captura de dependências de longo alcance.
- A autoatenção calcula uma soma ponderada dos valores (V), onde os pesos (ou atenção) são obtidos pela relação entre queries (Q) e keys (K). Quanto maior o peso, maior a relevância daquele elemento para o contexto.
- Para evitar que os pesos fiquem muito atenuados em sequências longas e para permitir que o modelo aprenda diferentes tipos de relações entre os dados, utiliza-se a atenção multicabeça ou **multi-head attention**:
    - múltiplas cabeças de atenção paralelas, cada uma projetada para subespaços dimensional diferente.
    - As saídas são concatenadas e projetadas linearmente, aumentando  os diferentes modos de visão e representação dos dados.  


### Estrutura do tensor  

- A estrutura do tensor segue o modelo padrão codificador/decodificador:
    - **O codificador** é uma pilha de camadas com duas subcamadas:
        - `autoatenção` - captura as relações entre as palavras
        - `feed foward` - uma camada pequena de rede neural usando ReLU que permite transformaçoes complexas não lineares
        - Entre essas subcamadas há uma conexão residual para transformações (feed foward)  e normalização (LayerNorm)
        - Fórmula (pré‑LayerNorm): `saída = LayerNorm(entrada + Autoatenção(entrada))` e `saída_final = LayerNorm(saída_parcial + FeedForward(saída_parcial))`.
    - **O decodificador** são uma sequencia de camadas iguais com 3 subcamadas:
        - `Camada de autoatenção mascarada` - igual ao codificador, mas com uma máscara que impede a autoatenção do decodificador de ver os tokens gerados aumentando a casualidade
        - `Atenção cruzada` - uma camada intermediária que pode consultar a saída do codificador para gerar saída, ela alinha a saída do decodificador com a saída original.
        - `feed foward` - identico ao codificador, com conexões residuais e LayerNorm
        - Existe uma máscara que impede a autoatenção do decodificador de ver os tokens gerados aumentando a casualidade
OBS no modelo original do paper havia o embeding, 6 camadas do codificador e 6 camadas do decodificador e com 512 dimensões de modelo dim_model.  

![Modelo do tensor](/bibliografia/imagens/tensor_0.png)

---
###  O que é atenção?  


A função atenção é uma consulta (Query) em um padrão Chave-Valor (Key-Value),  onde a saída é uma soma ponderada dos valores V dados pelos pesos  que são a compatibilidade da Consulta (Q) e a Chave (K).  


![Modelo do tensor](/bibliografia/imagens/tensor_1.png)


O tensor utiliza como fórmula o **produto escalar escalonado** com formula:  

$$
Att(K,Q,V) = softmax(\frac{Q * K^{t}}{\sqrt{d_k}})V
$$

O fator $\sqrt{d_k}$ **escala** o produto para evitar que os valores cresçam demais, evitando que a magnitude não prevaleça sobre os gradientes e torma os gradientes estáveis.  

Quando temos apenas uma cabeça, criamos apenas uma relação sobre um conjunto de dados. No multihead attention usamos **multiplas cabeças**, cada um com seu próprio Q, K, V buscando relações sobre diferentes perspectivas, por fim essas perspectivas são concatenadas e projetads linearmeste:  

$$MultiHead(K, Q, V) = Concatenar(head_1, ..., head_n)* W$$

Onde head é:  

$$head=Att(QW^q, KW^k, VW^v)$$

O custo computacional de usar cabeças paralelas tem valor parecido com o processamento de uma única cabeça com as dimensões totais de cabeças.  

---
### Feed foward no modelo do tensor
Até o momento o modelo do tensor sofreu apenas multiplicações de vetores, ou seja transformações lineares, o que torna o resultado da atenção um resultado linear que poderia ser simplificado em uma única transformação linear. Assim é preciso adicionar uma não linearidade ao modelo permitindo criar mais possibilidades. Para isso usamos uma rede feed foward completamente conectada com a saída do modelo de atenção e uma função ReLU.  

![Modelo do tensor](/bibliografia/imagens/tensor_2.png)

- A rede feed foward consiste em uma operação de pesos (W) e bias(b) de uma rede neural tradicional e usa a fórmula $FFN(x) = max(0, xW_1 + b_1) W_2 + b_2$ .
- A função ReLU consiste em zerar os valores negativos do resultado do feed foward, gerando valores esparsos, criando agregações (clusters), além de favorecer a generalização.  

#### Por que ReLU?
ReLU é uma operação computacionalmente barata `max(0, x)` que adiciona não linearidade e esparsidade (tende a zerar metade das ativações) diminuindo o risco de treinamento excessivo (overfitting), além de favorecer a generalização. A função ReLU, possui derivada 1 quando x>0, logo o gradiente se mantém constante se for positivo, assim evitamos perdas do gradiente a medida que aumentamos a quantidade de tensores.
Soft ReLU é computacionalmente mais cara e mantém um pequeno valor negativo o que reduz a esparsidade, a um custo computacional maior, o que pode melhorar o desempenho dependendo da situação. Hoje com o uso mais eficiente das CPU e GPU o custo da soft ReLU é menor, mas os ganhos ainda são pequenos, sendo uteis apenas em modelos maiores como  BERT, GPT-2/3. No fim cabe a você definir o melhor uso e aplicar de acordo com sua capacidade computacional.  


---
### Aplicando o embedding
O computador não entende palavras, ele entende apenas números, então as palavras são convertidas em números com dimensões fixas, no caso `dim_model=512` no artigo original. O embedding ocorre na **entrada do codificador**, convertendo o texto em um vetor de embedding e no **embedding de saída** que transforma os tokens recebidos em vetores (como no caso de tradução ou recebendo dados de outro tensor).  

![Modelo do tensor](/bibliografia/imagens/tensor_3.png)

Outro ponto importante onde se usa o embeding é na saída linear do output, que será melhor explicado [aqui.](/README.md#gerando-o-output)

---
### Codificação posicional
O tensor não possui uma ordenação na entrada dos dados, todos os dados são lidos e processados simultaneamente, assim precisamos de um meio de adicionar aos dados uma um posicionamento relativo. Tensores trabalham apenas com atenção e feed foward o que significa que a troca de posição de dados não altera o resultado, assim foi criado a codificação posicional.  

![Modelo do tensor](/bibliografia/imagens/tensor_4.png)

A codificação posicional pode ser feita de 2 modos:
- codificação aprendida, onde é treinado um modelo que cria uma matriz de posição (`max_len x dim_model`). Esta abordagem é simples porém não consegue generalizar para tamanhos maiores que o original.
- codificação fixa - é usada uma fórmula para calcular a posição. No artigo original é usada essa abordagem.
A formula de codificação fixa é:
$$

PE(pos, 2i)   = sin( pos / 10000^{2\frac{i}{dim_{model}}} )

$$
$$

PE(pos, 2i+1) = cos( pos / 10000^{2\frac{i}{dim_{model}}} )

$$
Sendo `pos` a posição na sequencia de tokens, `i` a dimensão do dim_model (i entre 0 e dim_model), `2i` e `2i+1` significa que as posições pares recebem seno e as ímpares cosseno.  
Com base nessa codificação o comprimento da onda seno e cosseno varia de `2π` na posição 0 a `10000*2π`. O valor 10000 foi arbitrário e pode ser usado outro valor.  

---
### Gerando o output  
Para a geração de saída de dados ocorre uma etapa de projeção linear da saída do tensor com a matriz de embedding, (tam_vocabulario, dim_model). Ao fazer `saida_tensor @ matriz_embedding.T` geramos um `logits` projeção entre o resultado do tensor e a matriz de embedding do nosso vocabulário, com formato (batch, compr_seq, tam_vocabulario). 

![Modelo do tensor](/bibliografia/imagens/tensor_5.png)

O logits por sua vez é aplicado uma função `softmax`, transformando-o em  uma matriz de probabilidades de de vocabulario. Assim podemos escolher o resultado que mais se aproxima de um token, normalmente, o valor de maior probabilidade, gerando assim uma lista de tokens de resposta.  

---
### Um pouco de prática

#### 1 - Implementação básica do tensor
- Aqui podemos ver a primeira etapa do tensor onde criamos um modelo básico com Q, K, V e O.
    - O código pode ser conferido [aqui](/doc/aprendendo_tensor/tensor_1.py), basta executar `python doc/aprendendo_tensor/tensor_1.py`.  

#### 2 - Implementação básica da atenção
- Prosseguindo com o modelo, aqui temos a propagação do resultado usando dados ficticios.
    - No fluxo normal o texto é convertido em tokens passa por um processo de embeding.
    - No embeding os tokens se tornam vetores multidimensionais com tamanho `(batch, tam_sequencia, dim)`:
        - `batch` é a quantidade de sequências de valores independentes que são processados simultaneamente, assim ao invés de processar 10 frases em um loop adiciona a dimensão batch para processar as frases simultaneamente
        - `tam_sequencia` é o numero de tokens usados simultaneamente em cada batch
        - `dim` é o a dimensionalidade de cada token usado no embeding
    - Após passar pelo tensor a representação do embeding se torna um novo vetor de mesmo formato, podendo passar por um novo tensor ou ser decodificado.
    - Por fim o vetor resultado pode passar por um decodificador gerando texto, um classificador ou mesmo ir para outro tensor
    - O código pode ser conferido [aqui](/doc/aprendendo_tensor/tensor_2.py), basta executar `python doc/aprendendo_tensor/tensor_2.py`.  

#### 3 - implementação básica do feed foward
- Apos implementar a atenção, inserimos o feed foward com a função ReLU.
    - Consiste em dois pesos e bias completamente conectados com a saída da atenção aplicando a função ReLU $FFN(x) = max(0, xW_1 + b_1) W_2 + b_2$ 
    - Também foi implementado outras funções de ativação caso queiram testar.
- O código pode ser conferido [aqui](/doc/aprendendo_tensor/tensor_3.py), basta executar `python doc/aprendendo_tensor/tensor_3.py`.  

#### 4 - implementado o embedding
- O passo a passo do código de embedding do codificador e decodificador é:
    - criado um embedding de dimensão (tam_bloco, compr_seq, dim_model) de valor (2, 10, 64)
    - aplica a codificação posicional para que a posição seja significativa [Não implementada ainda]
    - propaga no tensor usando Q, K, V = embedding de 4 cabecas
        - ao propagar é multiplicado o embedding em cada cabeça de dimensão (num_cabecas, dim_model, dim_head) no valor de (2, 64, 16). dim_head será dim_model//num_cabeças
        - isso resulta em uma lista (batch, seq_len, dim_head x num_cabecas) de valor (2, 10, 16 x 4)
        - o resultado é concatenado no ultimo eixo gerando uma saida (2, 10, 64)
        - por fim é calculado o output onde a saída concatenada é multiplicada pelo peso de saída: W_O (dim_head*num_head, dim_model) * saida concatenada = (num_batch, seq_len, dim_model) no valor de (2, 10, 16)
- O passo a passo do embedding de saída é:
    - O resultado do decodificador passa pela matriz de embedding original, atraves de uma multiplicação `saida_frase @ matriz_embedding.T` assim geramos um resultado (batch, compr_seq, tam_vocab) com valor (2, 10, 1000).
    - esse resultado passa pelo softmax gerando valores entre 0 e 1 ao longo do vetor (2, 10, 1000)
    - em seguida escolhemos o maior valor de probabilidade para cada unidade de batch (no caso 2) usando `np.argmax(probabilidades_tokens, axis=-1)`
    - como o batch é 2 escolhemos os 2 blocos de tamanho seq_len com maior probabilidade. Como no exemplo seq_len = 10 e batch = 2 temos 10 tokens em 2 blocos
- O código pode ser conferido [aqui](/doc/aprendendo_tensor/tensor_4.py), basta executar `python doc/aprendendo_tensor/tensor_4.py`.  

#### Positional encodig
- O modelo segue conforme o `tensor_4.py`, apenas adicionamos a fómula do posicional enconding e somamos o posicional encoding ao embedding
- O código pode ser conferido [aqui](/doc/aprendendo_tensor/tensor_5.py), basta executar `python doc/aprendendo_tensor/tensor_5.py`. 


