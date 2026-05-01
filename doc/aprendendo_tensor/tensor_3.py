import numpy as np

def softmax(x, axis=-1):
    """Aplica o softmax para normalizar os valores em valores entre 0 e 1."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def calcular_atencao(Q, K, V):
    """
    Aplica a formula da atenção escalar por meio de produto escalar.
    Q, K, V: tensores de dimensões (..., seq_len, d_k)
    Retorna: saída (..., seq_len, d_v) e pesos de atenção (..., seq_len, seq_len)
    """
    dim_k = Q.shape[-1] # pega a dimensão da cabeça
    #primeira parte da formula softmax(Q*K^T)
    # A transposição ocorre somente na dim_model e dim_k.
    # Isso resulta em uma matriz quadrada
    score_QK = (Q @ K.transpose(0,2,1))/ np.sqrt(dim_k)
    atencao_normalizada = softmax(score_QK, axis=-1)
    saida = atencao_normalizada @ V
    return saida, atencao_normalizada

class FeedFoward:
    def __init__(self, dim_model, dim_ff=None, teste=True):
        '''
        Modelo da rede neural feed foward simplificada para gerar aleaoriedade ao modelo
        Params:
            dim_model: dimensão original do modelo do tensor - quantos parametros são aceitos
            dim_ff: é a dimensão do feed foward
        '''
        if teste: #definindo uma seed fixa para o numpy gerar sempre o mesmo valor no random
            np.random.seed(42)

        # segundo o paper "Attention Is All You Need", dim_ff é 4*dim_model
        dim_ff =dim_ff if dim_ff else 4 * dim_model
        
        #gerando os pesos e bias do feed foward
        self.W1 = np.random.randn(dim_model, dim_ff) * 0.1
        self.b1 = np.zeros(dim_ff)

        # Perceba que no segundo peso é invertido dim_ff e dim_model
        # O que irá gerar uma matriz quadrada dim_model x dim_model
        self.W2 = np.random.randn(dim_ff, dim_model) * 0.1
        self.b2 = np.zeros(dim_model)
    
    def forward(self, saida_atencao):
        '''
        Método que aplica o feed foward a saída da atenção
            saida_atencao = valor que saiu da atencao (batch, seq_len, dim_model)
        '''
        # Aplicando a transformação do feed foward
        ff1 = saida_atencao @ self.W1 + self.b1

        # Basta trocar a função de ativação aqui
        camada_oculta = self.relu(ff1)
        return (camada_oculta @ self.W2) + self.b2
    
    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def gelu(x):
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

    @staticmethod
    def swish(x, beta=1.0):
        return x * (1 / (1 + np.exp(-beta * x)))

class Tensor:
    def __init__(self, dim_model:int, heads:int, dim_k:int=0, dim_v:int=0, teste=True):
        '''
        Classe principal que opera o tensor
        Params:
            dim_model: dimensão do modelo, ou quantos parâmetros de entrada ele aceita
            heads: numero de cabeças de atenção
            dim_k, dim_v: dimensões usadas na cabeça. Se 0 usa divisao inteira de dim_model // heads
        '''
        self.dim_model = dim_model
        self.cabecas = heads
        self.dim_K = dim_k if dim_k != 0 else dim_model // heads
        self.dim_V = dim_v if dim_v != 0 else dim_model // heads

        if teste: #definindo uma seed fixa para o numpy gerar sempre o mesmo valor no random
            np.random.seed(42)

        '''
        Definindo os pesos iniciais de Q, K, V e o peso final O após a concatenação obedecendo:
        MultiHead(Q,K,V ) = Concat(head1,...,headh)W^o
        '''
    
        self.W_K = np.random.randn(self.cabecas, self.dim_model, self.dim_K) * 0.1
        self.W_Q = np.random.randn(self.cabecas, self.dim_model, self.dim_K) * 0.1
        self.W_V = np.random.randn(self.cabecas, self.dim_model, self.dim_V) * 0.1

        # Projeção final após concatenação
        self.W_O = np.random.randn(self.cabecas * self.dim_V, self.dim_model)
    
    def propagar(self, embeding_Q, embeding_K, embeding_V):
        '''
        Após o texto de entrada ser transformado em tokens e gerado um embeding,
        ele é propagado no tensor para gerar os resultados.
        Propagar é usar um embeding para multiplicar os vetores do tensor 
        e assim gerar os resultados.
        Params:
            embeding_Q: valor da dimensão Q do embeding
            embeding_K: valor da dimensão K do embeding
            embeding_V: valor da dimensão K do embeding         
        '''
        
        saida_cabecas = []
        for h in range(self.cabecas): #propagando ao longo de cada cabeça
            # Projetar Q, K, V com as matrizes da cabeça i            
            Q_i = embeding_Q @ self.W_Q[h]   # (batch, seq_len, d_k)
            K_i = embeding_K @ self.W_K[h]   # (batch, seq_len, d_k)
            V_i = embeding_V @ self.W_V[h]   # (batch, seq_len, d_v)
            
            #Calcula a atenção para cada cabeça
            atencao_saida, _ = calcular_atencao(Q_i, K_i, V_i)
            saida_cabecas.append(atencao_saida)

        #concatena a saída de acordo com o útimo eixo, no caso as cabecas
        # Ao concatenar retorna ao shape original no caso (cabecas, dim_model, dim_K)
        saida_concatenada = np.concatenate(saida_cabecas, axis=-1)
        
        #Calcula a saída multiplicando pelo peso de saída
        saida = saida_concatenada @ self.W_O
        return saida
    
if __name__ == "__main__":
    cabecas = 4
    dim_model = 64

    #criando o tensor
    tensor = Tensor(dim_model, cabecas)

    tensor_shape = tensor.W_Q.shape
    print("======================================\n\tCódigo do Tensor_2")
    print("======================================")
    print(f'Formato do tensor: {tensor.W_Q.shape}:')
    print(f'\t- {tensor.W_Q.shape[0]} Cabeças')
    print(f'\t- {tensor.W_Q.shape[1]} dimensões de valores')
    print(f'\t- {tensor.W_Q.shape[2]} dimensões de cabeça')
    

    print("\n\n======================================\n\tAplicando a propagação")
    print("======================================")   
    #simulando o embeding
    tam_batch = 4 #batch são o total de blocos de textos processados paralelamente
    tam_bloco = 64
    embeding_Q = np.random.randn(tam_batch, tam_bloco, dim_model)
    embeding_K = np.random.randn(tam_batch, tam_bloco, dim_model)
    embeding_V = np.random.randn(tam_batch, tam_bloco, dim_model)    

    saida = tensor.propagar(embeding_Q,embeding_K,embeding_V)
    print("✅ Multi-Head Attention executada com sucesso!")    
    print("Embeding de entrada no formato: ", embeding_Q.shape)
    print("Formato do tensor na concatenação: ", tensor.W_K[0].shape)
    print("Cada cabeça trabalha em subespaços de dimensão dim_k =", tensor.dim_K)
    print("Concatenação das", cabecas, "cabeças produz", cabecas * tensor.dim_V, "dimensões")
    print("Projeção final reduz para", dim_model, "dimensões")
    print("Resultado final igual a ", saida.shape, "dimensões")

    
    print("\n\n======================================\n\tAplicando a feed foward")
    print("======================================")   
    ff = FeedFoward(dim_model)
    res_ff = ff.forward(saida)
    print("✅ Feed foward executada com sucesso!")
    print(f"Peso e bias 1 com {ff.W1.shape}, {ff.b1.shape} dimensões")
    print(f"Peso e bias 2 com {ff.W2.shape}, {ff.b2.shape} dimensões")
    print(f"Saída do feed foward com {res_ff.shape} dimensões")

