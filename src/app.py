from src.modelo.tokenizador.processadores_texto.TokenizadorBPE import TokenizadorBPE
from src.modelo.tokenizador.processadores_texto.tokenizador_word_piece import TokenizadorWordPiece
from src.modelo.tokenizador.tokenizador import Tokenizador
from src.modelo.embeding.embeding import embeding_teste
from src.modelo.transformer.transformer import Transformer

import numpy as np


def tokenizar():
    bpe = TokenizadorBPE().processar_textos()
    #word_piece = TokenizadorWordPiece().aplicar_word_piece()
    ''' tk = Tokenizador()
    tk.carregar_tokenizador_bpe()
    with open('src/media/dataset/3.txt', 'r', encoding='utf-8') as f:
        t = f.read()
        tks = tk.tokenizar(t)
        print(len(list(t)), len(tks),f'{(1- (len(tks)/len(list(t))))*100} de redução ')
        rev_tk = tk.reverter_tokens(tks)
        res = ''
        for i in rev_tk:
            res+=i
        print(len(list(t)), len(list(res)),(len(list(res))/len(list(t)))*100)'''

def embeding():
    embeding_teste()

def testar_transformer():
    dim_model = 50
    num_heads = 8
    
    x = np.array([[[0,0],[0,1],[1,0],[1,1]]])
    y = np.array([
            [[0,0],
            [0,0],
            [0,0],
            [1,1]]
         ])

    padding = dim_model - x.shape[-1]
    print(x.shape)
    x = np.pad(x, pad_width=((0,0), (0,0), (0, padding)), constant_values=0)
    
    y = np.pad(y, pad_width=((0,0), (0,0), (0, padding)), constant_values=0)

    print("====="*10)
    print("\tAplicando o transformer padrao")
    print("====="*10)
    t = Transformer(dim_model=dim_model, num_heads=num_heads)
    i=0
    while True:
        i+=1
        x_norm = t.aplicar_tensor_padrao(x,x,x)
        t.aplicar_backward_padrao(x_norm, y)    
        x_corrigido = t.aplicar_tensor_padrao(x,x,x)        
        similaridade = t.utils.similaridade(y, x_corrigido)
        
        if i%100==0:
            print("====="*10)
            print("\tAplicando o backward")
            print("====="*10)
            print('>>> Similaridade apos backward', similaridade)
            print(">>> Similaridade média", similaridade.mean())
        if similaridade[-1]>= 0.98:
            break
    print(f">>> resultado final em {i} steps: ")
