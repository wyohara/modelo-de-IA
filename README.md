
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
        - `feed foward` - uma camada pequena de rede neural usando ReLu que permite transformaçoes complexas não lineares
        - Entre essas subcamadas há uma conexão residual para transformações (feed foward)  e normalização (LayerNorm)
        - Fórmula (pré‑LayerNorm): `saída = LayerNorm(entrada + Autoatenção(entrada))` e `saída_final = LayerNorm(saída_parcial + FeedForward(saída_parcial))`.
    - **O decodificador** são uma sequencia de camadas iguais com 3 subcamadas:
        - `Camada de autoatenção mascarada` - igual ao codificador, mas com uma máscara que impede a autoatenção do decodificador de ver os tokens gerados aumentando a casualidade
        - `Atenção cruzada` - uma camada intermediária que pode consultar a saída do codificador para gerar saída, ela alinha a saída do decodificador com a saída original.
        - `feed foward` - identico ao codificador, com conexões residuais e LayerNorm
        - Existe uma máscara que impede a autoatenção do decodificador de ver os tokens gerados aumentando a casualidade
OBS no modelo original do paper havia o embeding, 6 camadas do codificador e 6 camadas do decodificador e com 512 dimensões de modelo dim_model.  


###  O que é atenção?

A função atenção é uma consulta (Query) em um padrão Chave-Valor (Key-Value),  onde a saída é uma soma ponderada dos valores V dados pelos pesos  que são a compatibilidade da Consulta (Q) e a Chave (K). O tensor utiliza como fórmula o **produto escalar escalonado** com formula:  

$$
Att(K,Q,V) = softmax(\frac{Q * K^{t}}{\sqrt{d_k}})V
$$

O fator $\sqrt{d_k}$ **escala** o produto para evitar que os valores cresçam demais, evitando que a magnitude não prevaleça sobre os gradientes e torma os gradientes estáveis.  

Quando temos apenas uma cabeça, criamos apenas uma relação sobre um conjunto de dados. No multihead attention usamos **multiplas cabeças**, cada um com seu próprio Q, K, V buscando relações sobre diferentes perspectivas, por fim essas perspectivas são concatenadas e projetads linearmeste:  

$$MultiHead(K, Q, V) = Concatenar(head_1, ..., head_n)* W$$

Onde head é:  

$$head=Att(QW^q, KW^k, VW^v)$$

O custo computacional de usar cabeças paralelas tem valor parecido com o processamento de uma única cabeça com as dimensões totais de cabeças.  

### Um pouco de prática
- Aqui podemos ver a primeira etapa do tensor onde criamos um modelo básico com Q, K, V e O.
    - O código pode ser conferido [aqui](/doc/aprendendo_tensor/tensor_1.py), basta executar `python doc/aprendendo_tensor/tensor_1.py`.  
- Prosseguindo com os modelos, aqui temos a propagação do resultado usando dados ficticios.
    - No fluxo normal o texto é convertido em tokens passa por um processo de embeding.
    - No embeding os tokens se tornam vetores multidimensionais com tamanho `(batch, tam_sequencia, dim)`:
        - `batch` é a quantidade de sequências de valores independentes que são processados simultaneamente, assim ao invés de processar 10 frases em um loop adiciona a dimensão batch para processar as frases simultaneamente
        - `tam_sequencia` é o numero de tokens usados simultaneamente em cada batch
        - `dim` é o a dimensionalidade de cada token usado no embeding
    - Após passar pelo tensor a representação do embeding se torna um novo vetor de mesmo formato, podendo passar por um novo tensor ou ser decodificado.
    - Por fim o vetor resultado pode passar por um decodificador gerando texto, um classificador ou mesmo ir para outro tensor
    - O código pode ser conferido [aqui](/doc/aprendendo_tensor/tensor_2.py), basta executar `python doc/aprendendo_tensor/tensor_2.py`.  



