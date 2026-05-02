import pandas as pd
from collections import defaultdict

UNK = '[?]'

class Tokenizador:
    def __init__(self):
        self.__arquivo_word_piece = 'src/media/dados_processados/lista_word_piece.csv'
        self.__arquivo_lista_bpe = 'src/media/dados_processados/lista_bpe.csv'
        self.__tokenizador = dict()
        self.__rev_tokenizador = dict()
    
    def carregar_tokenizador_word_piece(self, quantidade=120000):
        df = pd.read_csv(str(self.__arquivo_word_piece), encoding='utf-8')        
        self.__tokenizador = dict()
        self.__rev_tokenizador = dict()

        #ordena o resultado por quantidade do maior para o menor
        df = df.sort_values('freq', ascending=False)
        lista = df.values.tolist()[:quantidade]
        #subpalavra,quantidade,freq
        for i, tk in enumerate(lista):
            self.__tokenizador[tk[0]] = {'id': i+1}
            self.__rev_tokenizador[i+1] = {'token':tk[0]}
    
    def carregar_tokenizador_bpe(self, quantidade=120000):
        df = pd.read_csv(str(self.__arquivo_lista_bpe), encoding='utf-8')        
        self.__tokenizador = dict()
        self.__rev_tokenizador = dict()

        #ordena o resultado por quantidade do maior para o menor
        df = df.sort_values('quantidade', ascending=False)
        lista = df.values.tolist()[:quantidade]
        #subpalavra,quantidade,freq
        for i, tk in enumerate(lista):
            self.__tokenizador[tk[0]] = {'id': i+1}
            self.__rev_tokenizador[i+1] = {'token':tk[0]}
    
    def tokenizar(self, texto:str)->list:
        '''
        Metodo que percorre o texto e cria tokens
        Params:
            texto: texto a ser tokenizado
        '''
        #achando a maior subpalavra para o metodo guloso
        maior_chave = max(len(str(chave)) for chave in self.__tokenizador.keys())
        tkr = self.__tokenizador
        resposta = []
        #Realiza o loop enquanto o texto não estiver vazio
        while len(texto)>0:
            achado = False
            #usa o método guloso procurando do maior token para o menor
            for i in range(min(maior_chave, len(texto)), 0, -1):
                subpalavra = texto[:i]

                # se achar o token salva na resposta o id e marca como achado
                if subpalavra in tkr.keys():
                    resposta.append(tkr[subpalavra]['id'])
                    texto = texto[i:]
                    achado = True
                    break
            
            #ao sair do loop se a flag achado for False marca como UNK
            if not achado:
                resposta.append(UNK)
                texto=texto[1:]
        return resposta
    
    def reverter_tokens(self, lista_tokens:list):
        '''
        Metodo que percorre a lista de tokens e converte o id em subpalavra
        '''
        resultado = []
        for id in lista_tokens:
            #ignora o caractere UNK para gerar as subpalavras
            if id != UNK:
                resultado.append(self.__rev_tokenizador[id]['token'])
            else:
                resultado.append(UNK)
        return resultado