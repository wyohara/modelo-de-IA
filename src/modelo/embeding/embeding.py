import numpy as np
from src.modelo.tokenizador.tokenizador import Tokenizador
from src.modelo.tensor.modelo_personalizado.tensor import Tensor


def tensor_embeding(tensor, texto):
    dim_model = 300
    num_head = 50
    tkr = Tokenizador()

    tkr.carregar_tokenizador_bpe()
    texto_tokenizado = tkr.tokenizar(texto)        
    texto_reshape = tensor.converter_lista_para_tensor_shape(texto_tokenizado)
    att = tensor.mha.gerar_atencao(texto_reshape, texto_reshape, texto_reshape)
    return tensor.feedfoward.foward(att)

def embeding_teste():       
    dim_model = 512
    num_head = 12
    t = Tensor(seq_len=dim_model, num_heads=num_head)
    palavras = ['rei', 'rainha', 'arvore', 'mulher']
    
    t.similaridade(tensor_embeding(t, palavras[0]), tensor_embeding(t, palavras[1]))
    t.similaridade(tensor_embeding(t, palavras[0]), tensor_embeding(t, palavras[3]))

    rotulo = t.rotacionar_por_cosseno(tensor_embeding(t, palavras[0]), tensor_embeding(t, palavras[3]), 0.1)
    t.corrigir_tensor(tensor_embeding(t, palavras[0]), rotulo)
    t.similaridade(tensor_embeding(t, palavras[0]), tensor_embeding(t, palavras[3]))
    


