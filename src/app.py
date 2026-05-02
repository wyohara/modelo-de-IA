from src.modelo.tokenizador.TokenizadorBPE import TokenizadorBPE
from src.modelo.embeding.embeding import embeding_teste


def tokenizar():
    '''tk = Tokenizador()
    res = tk.tokenizar("estelionatário")
    print(res)
    print(tk.reverter(res))
    print(tk.reverter(res, concatenar=True))'''

    tk = TokenizadorBPE()
    tk.processar_textos()


def embeding():
    embeding_teste()



