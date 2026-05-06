
# Modelo básico de IA para estudo  

Este é um projeto de um transformer para modelo de inteligência artificial basico para estudo. O modelo usa como base um transformer básico inicialmente pensado segundo o artigo "Attention is all you need" [(aqui)](/bibliografia/Attention_is_all_you_need.pdf).  

## Sumário  
1. [Princípios Filosóficos](/README.md#princípios-filosóficos)
2. [O transformer](/README.md#o-tensor)
3. [Tokenizador](/doc/tokenizacao.md#sumário)

---
## Princípios Filosóficos 
Antes de pensar em inteligência artificial é preiso compreender o que ela é, entender seus limites e características e é para isso que este capítulo serve. Por definição nossa espécie é *Homo Sapiens* ou"humano sábio", pois nossa **inteligência** foi o fator determinante para a nossa sobrevivência. Foi por meio da nossa **inteligência** que percebemos o meio, o compreendemos, prevemos as possibilidades e criamos formas de manipular o mundo. 

Assim se seguissemos esse paralelo, poderíamos pensar que Inteligência Artificial é a replicação das mesmas capacidades dos humanos de enxergar, prever e criar possibilidades. Porém, atualmente, abrange uma série de subcampos e objetivos distintos tais como aprendizado, raciocínio ou mesmo atividades específicas como traduzir, criar código, jogar xadrez entre outros.
 
### A visão tradicional da inteligência artifical  
A forma mais tradicional de enxergar a inteligência artificial é por meio da ***racionalidade*** e ***humanidade***. Essas duas vertentes debatem se a inteligência artificial deve ser fiel ao humano ou ser simplificado para fazer o correto de forma racional, podendo ter 4 vertentes:  

#### 1. Agir como humano
Essa é a visão tradicional proposto por **Alan Turing (1950)**, ela foi proposta para preencher a ideia geral **"O humano pode pensar?"**. Nela o computador seria interrogado por meio de perguntas escritas e após responder o interrogador deveria dizer se foi respondido por uma pessoa ou computador. Para poder passar no teste o computador precisaria de alto dempenho em: 
- ***1 - Processamento da linguagem natural*** para poder se comunicar.
- ***2 - Raciocínio automatizado***, não necessariamente lógico, para responder e tirar conclusões
- ***3 - Armazenamento do conhecimento*** armazenar o que percebe e aprende em uma representação do conhecimento
- ***4 - Aprendizado de máquina*** para se adaptar aos padrões desconhecidos para generalizar e extrapolar.

Além das capacidades acima, a ciência moderna propôs a extrapolação para um ***Teste de Turing total***, onde o computador não se limita apenas a responder por texto, ele também percebe e interage com o ambiente, e a partir disso o interrogador define se as ações são de um humano ou máquina. Para essa atividade é preciso das capacidades:  
- ***5 - Visão Computacional***: Capacidade de perceber o mundo por outros meios além do texto
- ***6 - Robótica***: Para poder interagir fisicamente com o meio

Essas 6 capacidades compõem a maior parte dos campos da Inteligência artificial.  


#### 2. Pensar como humano
É a visão tradicional que busca entender como os humanos pensam. Podemos generalizar as técnicas em três metodos:
 - ***1 - Introspecção***: Capturar os pensamentos enquanto ocorrem;
 - ***2 - Experimentos psicológicos***: Observar as pessoas em ação
 - ***3 - Imagem Cerebral***: Observar o cérebro em ação

Esta ciência é bem ampla e com muitas possibilidades, mas não é abordado aqui por precisar de equipamentos específicos e amostragem.  

Alguns livros interessantes para a análise:
- *WILSON, Robert A.; KEIL, Frank C. (ed.). **The MIT encyclopedia of the cognitive sciences**. Cambridge: MIT Press, 1999.*  
- *NEWELL, Allen; SHAW, J. C.; SIMON, Herbert A. **Report on a general problem-solving program**. Santa Monica: RAND Corporation, 1961.*  

#### 3. Pensar racionalmente
O pensar racionalmente já é um conceito filosófico antigo, Aristósteles foi um dos primeiros a codificar o pensamento correto, usando raciocínios lógicos irrefutáveis encadeados - o ***silogismo***. Um exemplo cássico é o silogismo *'Sócrates é um homem e todos os homens são mortais e conclui que Sócrates é mortal.'*. A partir de então se iniciou o estudo do campo da ***lógica***.  
Até o século XIX se acreditou que a lógica poderia resolver qualquer problema solucionável. Prém existe uma limitação importante: Requer que o conhecimento do mundo esteja certo - condição raramente alcançada, tanto pela própria definição de certo, quanto por não conhecermos todas as regras e fatos possíveis.  
A falta da definição completa do certo é preenchida através da **probabilidade**, permitindo uma aproximação usando informações incertas, assim podemos criar um **pensamento racional ambrangente**. Apenas pensar racionalmente não torna o agente racional, para isso é preciso também definir uma ***ação racional a partir do pensamento racional***.  

#### 4. Agir racionalmente
Aqui surge a definição de ***Agente***, que é todo aquele que age. Assim um computador que executa uma ação é um agente. Porém agir não é tudo que esperamos do computador, esperamos que ele seja autônomo, podendo agir dentro do que se espera ao longo do tempo - se tornando um ***Agente Autônomo***.  
Aqui ocorre a extrapolação, além de fazer inferências corretas, também age de forma correta. Um exemplo seria: *"O fogão esquentar dentro dos limites está tudo bem, mas aquecer além do limite o agente age acionando o alarme de incêndio"*.  

### Modelo padrão
Quando definimos uma tarefa definida para um agente autonomo, como jogar xadrez ou fazer um cálculo, podemos usar um **modelo padrão**, isto é, um modelo simples em que é especificado todo o objetivo para o computador. Porém, a medida que nos aproximamos do mundo real, é cada vez mais difícil especificar todos os objetivos ou atingir o objetivo desejado, ou a resposta se afasta do desejo de quem solicitou. Um exemplo: se quisermos um carro autônomo com máxima segurança precisamos lidar com motoristas imprudentes, asfalto ruim, desgaste de equipamentos entre outos, tornando arriscado o deslocamento, assim a solução mais lógica não sair.  
O problema entre alinhar o objetivo desejado e o objetivo mais lógico se  chama ***problema de alinhamento de valores*** e quanto mais inteligente um agentemais difícil é alinhar esses valores, pois as consequências de uma correção se torna incalculável, imagine corrigir a segurança do mesmo carro, uma alteração nas características  de segurança pode fazer ele aceitar atropelar uma pessoa se os riscos de dano forem menores que o calculado, mesmo que esse cálculo decorra de uma falha instrumental.  

---
# O transformer  
No artigo “Attention Is All You Need” foi criada a proposta de um modelo de rede neural que substitui redes recorrentes (RNNs) e convolucionais (CNNs) por um mecanismo de atenção pura, chamado Transformer.
As principais características são: 
- O mecanismo de autoatenção que permite relacionar duas posições quaisquer da sequência de dados com custo constante, diferente de RNNs e CNNs que aumentavam o custo a medida que aumentavam a sequência de dados, facilitando a captura de dependências de longo alcance.  
- Para aplicar a autoatenção é calculada uma soma ponderada dos valores (V), onde os pesos (ou atenção) são obtidos pela relação entre consultas (Q) e chaves (K). Quanto maior o peso, maior a relevância daquele elemento para o contexto.  
- Para evitar que os pesos fiquem muito atenuados em sequências longas, pois quanto mais dados menor fica a diferença dos pesos, e para permitir que o modelo aprenda diferentes tipos de relações entre os dados, utiliza-se a atenção multicabeça ou multi-head attention:
    - múltiplas cabeças de atenção paralelas, cada uma projetada para subespaços dimensionais diferentes. Assim, um mesmo conjunto de dados é analisado por diferentes dimensões, cada um por uma cabeça.
Por fim as saídas são concatenadas e projetadas linearmente, aumentando os pesos de atenção por meio de diferentes modos de visão e representação dos dados.  

## Estrutura do Transformer 
A estrutura do transformer segue o modelo padrão codificador/decodificador, onde:  
- O codificador é uma pilha de camadas com duas subcamadas:  
    - autoatenção - captura as relações entre as palavras
    - Camada de adição e normalização - para evitar a perda do gradiente
    - feed foward - uma camada pequena de rede neural usando ReLU que permite transformações complexas não lineares, adicionando aleatoriedade ao modelo.
    - Camada de adição e normalização - para evitar a perda do gradiente
OBS A camada de adição e normalização é a adição e normalização dos valores originais da consulta com o resultado do gradiente, seguido de uma normalização. Isso evita a perda do gradiente (pesos e bias de cada parâmetro de entrada)  

$
saida_{att} = LayerNorm(entrada + Autoatenção(entrada))
resposta = LayerNorm(saida_{att}  + FeedForward(saida_{att}))
$

O decodificador são uma sequência de camadas iguais com 3 subcamadas:  
- Camada de autoatenção mascarada - igual ao codificador, mas com uma máscara que impede a autoatenção do decodificador de ver os tokens gerados aumentando a casualidade  
- Atenção cruzada - uma camada intermediária que pode consultar a saída do codificador para gerar saída, ela alinha a autoatenção do codificador com a saída original.  
- Camada residual e LayerNorm com adição de normalização  
- Feed forward - idêntico ao codificador
- Camada residual e LayerNorm com adição de normalização  

OBS No modelo original do artigo havia um embedding, 6 camadas do codificador e 6 camadas do decodificador e com 512 dimensões de modelo.  
 

![Modelo do transformer](/bibliografia/imagens/tensor_0.png)

---
###  O que é atenção?  


A função atenção é uma consulta (Query) em um padrão Chave-Valor (Key-Value),  onde a saída é uma soma ponderada dos valores V dados pelos pesos  que são a compatibilidade da Consulta (Q) e a Chave (K).  


![Modelo do transformer](/bibliografia/imagens/tensor_1.png)


O Transformer utiliza como fórmula o **produto escalar escalonado** com formula:  

$$
Att(K,Q,V) = softmax(\frac{Q * K^{t}}{\sqrt{d_k}})V
$$

Onde a função softmax é dada por:  

$$
\sigma(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}, \quad i = 1,\dots,K
$$  


> A função **softmax**   recebe como entrada um vetor z de K números reais e o normaliza em uma distribuição de probabilidade, onde a soma dessa distribuição retorna valor 1 e são proporcionais aos exponencial $e^{z_i}$ dos números de entrada. Ou seja, antes da softmax, alguns componentes do vetor podem ser negativos ou maiores que 1, além de não somar 1, mas, após aplicar a softmax, cada componente estará entre 0 e 1 e os componentes serão somados a 1, de modo que possam ser interpretados como probabilidades, além disso, os componentes de entrada maiores corresponderão a probabilidades maiores. [- Fonte](https://pt.wikipedia.org/wiki/Fun%C3%A7%C3%A3o_softmax)


O fator $\sqrt{d_k}$ **escala** o produto para evitar que os valores cresçam demais, evitando assim que a magnitude (comprimento ou intensidade) não prevaleça sobre os gradientes, além de tornar os gradientes estáveis.  

Quando temos apenas uma cabeça, criamos apenas uma relação sobre um conjunto de dados. Na atenção multicabeça (multihead attention) usamos **multiplas cabeças**, cada um com seus próprios valores de Q, K, V e buscando relações sobre diferentes perspectivas, por fim essas perspectivas são concatenadas e projetads linearmente:  

$$
MultiHead(K, Q, V) = Concatenar(head_1, ..., head_n)* W
$$

Onde cada cabeça (head) é:  

$$
head=Att(QW^q, KW^k, VW^v)
$$

Ao fim da concatenação um Tensor Q, K, V de formato `(batch, heads, seq_len, dim_head)` ao ter as cabeças concatenadas se torna um Tensor `(batch, seq_len, dim_model)` onde  `dim_model = heads*dim_head`.  
- `heads ou h`: é o número de cabeças do Transformers
- `seq_len`: é a quantidade de valores de entrada do modelo, como por exemplo o número de tokens de entrada.
- `dim_head dado por d_k e d_v`: é o número de dimensões que cada cabeça tem. Originalmente calculado como `d_k,d_v = dim_model/heads`
- `d_model` é o número de dimensões que o modelo suporta, no caso `heads*dim_head`.
    No artigo original havia `dim_head=8`, `d_model=512` e portanto `dim_head=64` 
- `Tensor` em algebra relacional é o nome dado a um vetor qualquer com mais de 3 dimensões.
- concatenar é o processo de unir vetores ou tensores, assim um vetor $A = [1, 2]$ e o vetor $B = [3, 4]$, aao concatenar resulta em $(A + B) = [1, 2, 3, 4]$.

Obs O custo computacional de usar cabeças paralelas tem valor parecido com o processamento de uma única cabeça com as dimensões totais de cabeças.  

---
### Feed Foward no modelo do transformer
Até o momento o modelo do transformer sofreu apenas transformações lineares que podem ser definidas no formato $A = Bx+c$, ou seja, transformações lineares que poderiam ser simplificadas em uma única transformação linear. Assim é preciso adicionar não linearidade ao modelo, permitindo criar mais resultados além de permitir uma generalização. Para isso usamos uma rede feed foward simples completamente conectada com a saída do modelo de atenção e uma função de ativação ReLU.  

![Modelo do transformer](/bibliografia/imagens/tensor_2.png)

- A rede **feed foward** consiste em uma rede neural em que os dados fluem apenas em uma direção e a informação de entrada sofre operações com pesos (W) e bias(b) com a fórmula $FFN(x) = (x \cdot W) + b$.
    Na rede neural original do transformer são utilizados apenas duas camadas de pesos e bias no formato $FFN(x) = ReLU[(x \cdot W_1 + b_1)\cdot W_2+b_2]$ seguido de uma ativação pela função ReLU.  
- A função ReLU consiste em uma função de ativação onde zeram os valores negativos e propagam os positivos. Ela é dada por $ReLU = max(0, x)$.  

#### Por qual motivo usar ReLU?
ReLU é uma operação computacionalmente barata `relu = max(0, x)` que adiciona não linearidade e esparsidade (tende a zerar metade das ativações), diminuindo o risco de treinamento excessivo (overfitting), além de favorecer a generalização.  
- Como contrapartida a Função ReLU tende a degradar o gradiente de propagação, criando uma tendencia de zerar os gradientes, sendo necessário assim a camada de adição e normalização dos dados, evitando a perda do gradiente.  
- A função ReLU, possui apenas a derivada 1 quando x>0, logo o gradiente se mantém constante caso for positivo, assim evitamos perdas do gradiente positivo a medida que aumentamos a quantidade de camadas no transformer.
- Outra opção é o `Soft ReLU`, que é computacionalmente mais cara, porém mantém um pequeno valor negativo, a um custo computacional maior.
    - Manter um pequeno valor negativo reduz a esparsidade, o que pode melhorar o desempenho dependendo da situação.
    - Com o uso mais eficiente das CPU e GPU o custo da soft ReLU é menor, mas os ganhos ainda são pequenos, sendo uteis apenas em modelos maiores como  BERT, GPT-2/3
    - No fim cabe a você definir o melhor uso e aplicar de acordo com sua capacidade computacional.  


---
### Aplicando o embedding
O computador não entende palavras, ele entende apenas números, portanto as palavras precisam ser convertidas em indices inteiros, os tokens (ex 'eu' = 1, 'ele'=2). A quantidade de tokens é dado por`seq_len` e tem tamanho variável e depende do modelo.  
- Após a tokenização, cada token é transformado em um vetor denso na **camada de embedding**, gerando uma matriz densa `(seq_len, d_model)` que será usado:  
- O Tensor de embedding é usado nos seguintes momentos: 
    - Na **entrada do codificador**
    - Na **entrada do decodificador** após o deslocamente a direita `shift right`

#### Shift Right
`shift right` é o processo de treinamento do decodificador onde ele aprende a gerar respostas, para isso ele é treinado com uma saída esperada do seguinte modo:
- é fornecido uma resposta alvo `ex ['a', 'vida', 'e', 'bela']` e uma entrada ocultando o ultimo elemento (shift right) `ex ['a', 'vida', 'e']`
- Espera-se que o decodificador gere a resposta com o último elemento e exclua o primeiro `ex ['vida', 'e', 'bela']`
- Com isso treinamos o decodificador a gerar sequencialmente as respostas sem saber o resultado. 
- além do sift`right é usado a mascara causal e mascara de padding.

#### Máscara causal (look‑ahead mask)
Para evitar que o decodificador saiba quais são os pesos  

![Modelo do transformer](/bibliografia/imagens/tensor_3.png)


---
### Codificação posicional
O transformer não possui uma ordenação na entrada dos dados, todos os dados são lidos e processados simultaneamente, assim precisamos de um meio de adicionar aos dados uma um posicionamento relativo. transformeres trabalham apenas com atenção e feed foward o que significa que a troca de posição de dados não altera o resultado, assim foi criado a codificação posicional.  

![Modelo do transformer](/bibliografia/imagens/tensor_4.png)

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
Para a geração de saída de dados ocorre uma etapa de projeção linear da saída do transformer com a matriz de embedding, (tam_vocabulario, dim_model). Ao fazer `saida_tensor @ matriz_embedding.T` geramos um `logits` projeção entre o resultado do transformer e a matriz de embedding do nosso vocabulário, com formato (batch, compr_seq, tam_vocabulario). 

![Modelo do transformer](/bibliografia/imagens/tensor_5.png)

O logits por sua vez é aplicado uma função `softmax`, transformando-o em  uma matriz de probabilidades de de vocabulario. Assim podemos escolher o resultado que mais se aproxima de um token, normalmente, o valor de maior probabilidade, gerando assim uma lista de tokens de resposta.  

---
### Um pouco de prática

#### 1 - Implementação básica do transformer
- Aqui podemos ver a primeira etapa do transformer onde criamos um modelo básico com Q, K, V e O.
    - O código pode ser conferido [aqui](/doc/aprendendo_tensor/tensor_1.py), basta executar `python doc/aprendendo_tensor/tensor_1.py`.  

#### 2 - Implementação básica da atenção
- Prosseguindo com o modelo, aqui temos a propagação do resultado usando dados ficticios.
    - No fluxo normal o texto é convertido em tokens passa por um processo de embeding.
    - No embeding os tokens se tornam vetores multidimensionais com tamanho `(batch, tam_sequencia, dim)`:
        - `batch` é a quantidade de sequências de valores independentes que são processados simultaneamente, assim ao invés de processar 10 frases em um loop adiciona a dimensão batch para processar as frases simultaneamente
        - `tam_sequencia` é o numero de tokens usados simultaneamente em cada batch
        - `dim` é o a dimensionalidade de cada token usado no embeding
    - Após passar pelo transformer a representação do embeding se torna um novo vetor de mesmo formato, podendo passar por um novo transformer ou ser decodificado.
    - Por fim o vetor resultado pode passar por um decodificador gerando texto, um classificador ou mesmo ir para outro transformer
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
    - propaga no transformer usando Q, K, V = embedding de 4 cabecas
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


