from src.modelo.tokenizador.processadores_texto.TokenizadorBPE import TokenizadorBPE
from src.modelo.tokenizador.processadores_texto.tokenizador_word_piece import TokenizadorWordPiece
from src.modelo.tokenizador.tokenizador import Tokenizador
from src.modelo.embeding.embeding import embeding_teste


def tokenizar():
    #bpe = TokenizadorBPE().processar_textos()
    #word_piece = TokenizadorWordPiece().aplicar_word_piece()
    tk = Tokenizador()
    tk.carregar_tokenizador_bpe()
    with open('src/media/dataset/3.txt', 'r', encoding='utf-8') as f:
        t = f.read()
        tks = tk.tokenizar(t)
        print(len(list(t)), len(tks),f'{(1- (len(tks)/len(list(t))))*100} de redução ')
        rev_tk = tk.reverter_tokens(tks)
        res = ''
        for i in rev_tk:
            res+=i
        print(len(list(t)), len(list(res)),(len(list(res))/len(list(t)))*100)


def embeding():
    embeding_teste()



