
# Modelo básico de IA para estudo  

Este é um projeto de um tensor para modelo de inteligência artificial basico para estudo. O modelo usa como base um tensor básico inicialmente pensado segundo o artigo "Attention is all you need" [(aqui)](/bibliografia/Attention_is_all_you_need.pdf).  

## Sumário  
1. [Princípios Filosóficos](/README.md#princípios-filosóficos)
2. [O tensor](/README.md#o-tensor)
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


