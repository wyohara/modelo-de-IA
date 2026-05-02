from pathlib import Path
import pandas as pd
from collections import defaultdict

class TokenizadorWordPiece:
    def __init__(self):
        self.__arquivo_lista_bpe = Path('src/media/dados_processados/lista_bpe.csv')
        self.__arquivo_lista_word_piece = Path('src/media/dados_processados/lista_word_piece.csv')

    def aplicar_word_piece(self):
        '''
        Método que calula o BPE e usa para ca calcular as requencias relativas.
        Usa a formula freq=num_ab/(num_a*num_b) onde num_ é a quantidade de ocorrencias 
        '''
        # Tenta ler o arquivo bpe existente e transformar em dicionário
        df_existente = pd.read_csv(str(self.__arquivo_lista_bpe), encoding='utf-8')
        lista_word_piece = defaultdict(dict)
        for _, row in df_existente.iterrows():
            lista_word_piece[row['subpalavra']] = {'quantidade': row['quantidade']}
        
        #percorre o dicionário
        print(f">>>>> Iniciando o word piece a partir do {self.__arquivo_lista_bpe.name}")
        for c in lista_word_piece.keys():

            # subpalavras de 1 caractere tem frequencia 1 para sempre ocorrer 
            if len(str(c))>1:
                #Como o BPE sempre concatena a direita a f_b é sempre a ultima letra
                c_a = str(c)[:-1]
                c_b = str(c)[-1:]
                try:
                    #aplicando a fórmula freq=num_ab/(num_a*num_b)
                    if c_a in lista_word_piece.keys() and c_b in lista_word_piece.keys():
                        num_ab = lista_word_piece[c]['quantidade']
                        num_a = lista_word_piece[c_a]['quantidade']
                        num_b = lista_word_piece[c_b]['quantidade']
                        freq = num_ab/(num_a*num_b)
                    else:
                        raise KeyError
                except KeyError:
                    freq = 0
            else:
                freq = 1
            lista_word_piece[c]['freq']= freq
        
        self.__salvar_word_piece(lista_word_piece)

    
    def __salvar_word_piece(self, lista_word_piece:defaultdict):
        '''
        Método que salva as frequencias de subpalavras que existam, substitui o arquivo anterior
        Params:
            lista_bpe: defaultdict contendo os dados a serem salvos
        '''
        arquivo_csv = str(self.__arquivo_lista_word_piece)

        # Converte o defaultdict/dict para DataFrame
        df_novo = pd.DataFrame([
            {'subpalavra': k, 'quantidade': v['quantidade'], 'freq':v['freq']}
            for k, v in lista_word_piece.items()
        ])

        #ordena o resultado por frequencia do maior para o menor
        df_novo = df_novo.sort_values('freq', ascending=False)
        
        df_novo.to_csv(arquivo_csv, index=False, encoding='utf-8')
        