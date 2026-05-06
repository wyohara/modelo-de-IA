from src.modelo.tokenizador.processadores_texto.TokenizadorBPE import TokenizadorBPE
from src.modelo.tokenizador.processadores_texto.tokenizador_word_piece import TokenizadorWordPiece
from src.modelo.tokenizador.tokenizador import Tokenizador
from src.modelo.embeding.embeding import embeding_teste
from src.modelo.tensor.modelo_personalizado.tensor import Tensor
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


def testar_tensor():
    dim_model = 2
    cabecas = 1
    
    x = np.array([[[1,2],[4,6],[7,3]]])
    print(x.shape[-1])
    y = np.array([[[3,7 ],
  [-0.9999995,   0.9999995 ],
  [ 0.99999948, -0.99999948]]])

    t=Tensor(seq_len=dim_model, num_heads=cabecas, teste=True)
    ff_soma, ff_out, mean, std, out_norm, gamma = t.mha.gerar_atencao_completa(x,x,x)
    print('>>> Saida da atencao - out', ff_out)
    print('>>> Saida da atencao - mean', mean)
    print('>>> Saida da atencao - std', std)
    print('>>> Saida da atencao - out norm', out_norm)
    print('>>> Saida da atencao - gamma', gamma)


    ff_soma, ff_out, ff_mean, ff_std, ff_out_norm, gamma= t.feedfoward.forward_completo(ff_out)
    print('>>> Saida da feed foward - ff out', ff_out_norm)
    print('>>> Saida da feed foward - ff mean', ff_mean)
    print('>>> Saida da feed foward - ff std', ff_std)
    print('>>> Saida da feed foward - ff norm', ff_out_norm)
    print('>>> Saida da feed foward - ff gamma', gamma)

    res_mse = t.feedfoward.gerar_gradiente_perda_mse(ff_out, y)
    print('MSE', res_mse)
    dx, dW1, db1, dW2, db2, dgamma, dbeta = t.feedfoward.gerar_backward(res_mse, ff_mean, ff_std, ff_out_norm, gamma, ff_out, ff_soma )

    print('antes correção', t.feedfoward.gamma, dgamma)
    t.feedfoward.corrigir_pesos(dW1,db1,dW2,db2, dgamma, dbeta)
    
    print('apos correção', t.feedfoward.gamma, t.feedfoward.beta)
    
    print('similaridade (cosseno) =', t.similaridade(x, y))
    rot = t.rotacionar_por_cosseno(x, y, 0.9)

    x_entrada,dW_Q_list, dW_K_list, dW_V_list,dW_O, db_O,dgamma_ln, dbeta_ln = t.mha.gerar_backward(dx, ff_out, mean, std, out_norm, gamma)

    #x_entrada vai para o tensor anterior

    print (db_O)


def testar_transformer():
    dim_model = 2
    num_heads = 1
    
    x = np.array([[[1,2],[4,6],[7,3]]])
    print(x.shape[-1])
    y = np.array([
            [[-0.9999995,  0.9999995 ],
            [-0.9999995,   0.9999995 ],
            [ 0.9999995, -0.9999995]]
         ])


    print("====="*10)
    print("\tAplicando o transformer padrao")
    print("====="*10)
    t = Transformer(dim_model=dim_model, num_heads=num_heads)
    resp = t.aplicar_tensor_padrao(x,x,x)
    print('>>> Saida do transformer', resp.shape) 
    print(resp)   
    print('>>> Similaridade', t.utils.similaridade(y, resp))

    