import numpy as np
from src.modelo.tokenizador.tokenizador import Tokenizador
from src.modelo.tensor.modelo_personalizado.tensor import Tensor


def tensor_embeding(texto):
    dim_model = 300
    num_head = 50
    tkr = Tokenizador()
    tensor = Tensor(seq_len=dim_model, num_heads=num_head, teste=False)

    tkr.carregar_tokenizador_bpe()
    texto_tokenizado = tkr.tokenizar(texto)        
    texto_reshape = tensor.converter_lista_para_tensor_shape(texto_tokenizado)
    att = tensor.mha.gerar_atencao(texto_reshape, texto_reshape, texto_reshape)
    return tensor.feedfoward.foward(att)

def embeding_teste():       
    dim_model = 300
    num_head = 50
    tensor = Tensor(seq_len=dim_model, num_heads=num_head)
    palavras = ['rei', 'rainha', 'arvore', 'mulher']
    
    tensor.similaridade(tensor_embeding(palavras[0]), tensor_embeding(palavras[1]))
    tensor.similaridade(tensor_embeding(palavras[0]), tensor_embeding(palavras[3]))
    


