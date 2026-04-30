
# Modelo básico de IA para estudo  

Este é um projeto de um tensor para modelo de inteligência artificial basico para estudo. O modelo usa como base um tensor básico inicialmente pensado segundo o artigo "Attention is all you need" [(aqui)](/bibliografia/Attention_is_all_you_need.pdf).  

## O Tensor  

Segundo o artigo foi proposto um modelo de tensor simplificando o modelo de rede neural e a otimizando. Suas principais características são:  
- Modelos convolucionais e redes recorrentes, tem em comum a implementação de um modelo de atenção, então foi proposto um modelo focado em atenção, chamado **tensor**.
- Ele ajuda a corrigir o problema de relacionar sinais distantes que possuem muita relação, o que antes era custoso passa a ser mais fácil.
    - Para isso é utilizado o mecanismo de autoatenção onde faz uma média ponderada de um valor com os demais valores a fim de determinar as relações próximas (valores maiores). Porém isso tem um custo.
    - Quanto mais sinais são usados nos tensores, mais diluída se torna a média ponderada. Para isso foi criado o **multi-head attention (atenção multicabeça)** onde cada cabeça aprende as relações de forma diferente e no fim são concatenadas linearmente, melhorando a diluição da média ponderada.  


### Estrutura do tensor  

- A estrutura do tensor segue o modelo padrão codificador/decodificador:
    - **O codificador** é uma pilha de camadas com duas subcamadas:
        - Camada de atenção - captura as relações entre as palavras
        - feed foward ou propagação - uma camada pequena que processa todas as palavras individualmente que ajuda a evitar a degradação do sinal
        - Entre essas subcamadas há uma normalização
        - Com essa estrutura o codificador realiza: ``saída parcial = LayerNorm(entrada + Autoatenção(entrada))` e `saída_final = LayerNorm(saída_parcial + FeedForward(saída_parcial))`.  
    - **O decodificador** são uma sequencia de camadas iguais com 3 subcamadas:
        - Camada de autoatenção igual ao codificador
        - Atenção cruzada - uma camada intermediária que pode consultar o resultado do codificador para gerar saída
        - feed foward - identico ao codificador
        - A um bloqueio impedindo que os dados cruzados do codificador alterem o decodificador
OBS no modelo original do paper havia o embeding, 6 camadas do codificador e 6 camadas do decodificador com 512 dimensões.  


###  O que é atenção?

A função atenção é uma consulta (Query) em um padrão Chave-Valor (Key-Value),  onde a saída é uma ponderação dos valores da Consulta (K) e a Chave (K). Tradicionalmente a atenção era feita pro produto escalar escalonado com muitas consultas e chaves empacotadas em uma função softmax. Porém isso pode impactar a magnitude, devido a multiplicação, então reduzimos a magnitude dividindo por $\sqrt{d_k}$ que é o total de dimensão das chaves:  

$$Att(K,Q,V) = \frac{softmax(Q * K^{t})}{\sqrt{d_k}}V$$

No multihead attention buscamos fazer uma relação linear onde aprendemos de diferentes perspectivas e concatenamos o resultado de Q e K e por fim concatenamos com V e ao fim aplicamos os pesos W:  

$$MultiHead(K, Q, V) = Concatenar(head_1, ..., head_n)* W$$

Onde head é:  

$$head=Att(QW^q, KW^k, VW^v)$$

O custo computacional de usar cabeças paralelas tem valor parecido com o processamento de uma única cabeça com as dimensões totais de cabeças.  

### Um pouco de prática
- Aqui podemos ver a primeira etapa do tensor onde criamos um modelo básico com Q, K, V e O: [aqui](/doc/aprendendo_tensor/tensor_1.py), basta executar `python doc/aprendendo_tensor/tensor_1.py`.  
- Prosseguindo com os modelos, aqui temos a propagação do resultado usando dados ficticios.
    - No fluxo normal o texto é convertido em tokens passa por um processo de embeding.
    - No embeding os tokens se tornam vetores multidimensionais com tamanho `(batch, tam_sequencia, dim)`:
        - `batch` é a quantidade de sequências de valores independentes que são processados simultaneamente, assim ao invés de processar 10 frases em um loop adiciona a dimensão batch para processar as frases simultaneamente
        - `tam_sequencia` é o numero de tokens usados simultaneamente em cada batch
        - `dim` é o a dimensionalidade de cada token usado no embeding
    - Após passar pelo tensor a representação do embeding se torna um novo vetor de mesmo formato, podendo passar por um novo tensor ou ser decodificado.
    - Por fim o vetor resultado pode passar por um decodificador gerando texto, um classificador ou mesmo ir para outro tensor
    - O código pode ser conferido [aqui](/doc/aprendendo_tensor/tensor_2.py), basta executar `python doc/aprendendo_tensor/tensor_2.py`.  



