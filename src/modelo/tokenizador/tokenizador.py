from src.modelo.tokenizador.modelos_processamentos.processador_bpe import Processador_BPE
from collections import defaultdict

class Tokenizador():
    def __init__(self, num_tokens=120000):
        self.__tokenizador = defaultdict()
        self.__rev_tokenizador = defaultdict()
        self.__maior_bloco = 0
        self.gerar_tokenizador(num_tokens=num_tokens)
        self.__desconhecido = '[?]'

    def gerar_tokenizador(self, num_tokens=120000):
        self.__tokens = Processador_BPE().get_tokens()[:num_tokens]
        for tk in self.__tokens:
            self.__tokenizador[tk[0]] = tk[1]
            self.__rev_tokenizador[tk[1]] = tk[0]

            #marcando o maior bloco para tokenizar
            if len(tk[0])>self.__maior_bloco:
                self.__maior_bloco = len(tk[0])
    

    def tokenizar(self, texto:str)->tuple[int]:
        set_keys = set(self.__tokenizador.keys())
        resultado = []
        texto = str(texto.strip())
        while(len(texto)>0):
            if texto:
                for i in range(len(texto),0,-1):
                    subpalavra = str(texto[:i])
                    if subpalavra in set_keys:
                        resultado.append(self.__tokenizador[subpalavra])
                        texto = texto.replace(subpalavra, '')
                        break
            
        return tuple(resultado)       

    def reverter(self, lista_tokens:tuple[int], concatenar:bool=False)->str:
        resultado = []
        for res in lista_tokens:
            try:
                resultado.append(self.__rev_tokenizador[res])
            except KeyError:
                resultado.append(self.__desconhecido)
        if concatenar:
            resultado_conc = ''
            for res in resultado:
                resultado_conc+=res
            return resultado_conc
        else: return tuple(resultado)