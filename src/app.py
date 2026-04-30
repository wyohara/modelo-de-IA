from src.modelo.tokenizador.processador_textos import ProcessadorTextos
from src.modelo.tokenizador.tokenizador import Tokenizador
from src.modelo.embeding.embeding import embeding_teste

def processar_arquivos():
    ProcessadorTextos().processar_arquivos_dataset()


def tokenizar():
    tk = Tokenizador()
    res = tk.tokenizar("estelionatário")
    print(res)
    print(tk.reverter(res))
    print(tk.reverter(res, concatenar=True))


def embeding():
    embeding_teste()



