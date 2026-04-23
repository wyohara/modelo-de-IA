from pathlib import Path
from collections import defaultdict
import pandas as pd
import time
import ast

from src.ferramentas.ferramentas import texto_para_hex, hex_para_texto

class Processador_BPE:
    def __init__(self):
        self.__lista_tokens = Path('src/media/dados_processados/tokens_bpe.csv')
        self.__savepoint = Path('src/media/dados_processados/savepoint_bpe.csv')

    def __contar_caracteres(self, path:Path)->defaultdict:
        '''
        Método que conta os caracteres do texto e salva em um default dict no formato: {chave:valor}
            Params:
                path: caminho até o arquivo
            Return:
                defaultdict: dicionário dos caracteres
        '''
        lista_bpe = defaultdict()
        with open(str(path), encoding='utf-8') as f:
            texto = f.read()
            for i in texto:
                try:
                    lista_bpe[i]+=1
                except KeyError:
                    lista_bpe[i]=1
        return lista_bpe

    def __achar_caractere_coringa(self, lista_char:defaultdict)->str:
        '''
        Método que percorre todas as possibilidades do utf-8 para achar um caractere não usado no texto.
        Esse caractere será usado de curinga para marcar o BPE.
        Params:
            lista_char:defaultdict = dicionário com os caracteres e contagens
        Return:
            str: caractere não usado no texto
        '''
        for codigo in range(32, int(0x10FFFF), 1):
            caractere = chr(codigo)
            if caractere not in lista_char.keys() and caractere not in [' ', '\n']:
                return caractere

    def __aplicar_bpe(self,lista_bpe:defaultdict, path:Path, caractere_chave:str, coringa:str):
        '''
        Método principal que aplica o BPE: Carrega o texto e substitui o token mais comum pelo curinga
        em seguida conta suas ocorrencias do coringa e salva o o token+caractere seguinte
            Params:
                path:Path = Caminho do texto
                caractere_chave:str = caractere a ser substituido
                coringa:str = caractere coringa a ser usado
        '''
        with open(str(path), encoding='utf-8') as f:
            texto = f.read()

            #substitui o token pelo coringa
            texto = texto.replace(str(caractere_chave), coringa)

            #percorre o texto procurando o coringa
            # O -1 para não estourar o tamanho do texto
            for i in range(len(texto)-1):
                if texto[i] == coringa:
                    #caso ache o coringa, pega o token original e adiciona o próximo caractere no dicionário
                    try:
                        lista_bpe[str(caractere_chave)+str(texto[i+1:i+2])] += 1
                    except KeyError:
                        lista_bpe[str(caractere_chave)+str(texto[i+1:i+2])] = 1
        
        return lista_bpe
    
    def __carregar_tokens_csv(self) -> pd.DataFrame:
        '''
        Métpdo acessório que carrega o dataframe no formato [chave/indice, valor]
            Return:
                pd.DataFrame = dataframe dos dados
        '''
        try:
            df = pd.read_csv(str(self.__lista_tokens),index_col=0)
        except Exception as e:
            df = pd.DataFrame(columns=['valor'])
        return df

    def __salvar_tokens_csv(self, dados:list):
        '''
        Método acessório que salva a lista de tokens em csv
            Params:
                dados:list = lista de dados a ser salvo no csv no formato [(chave, valor)]
        '''
        df = pd.DataFrame(columns=['valor'])
        #separa a coluna valor
        coluna_valor = df.columns[0]
        #itera os dados para salvar os valores
        for chave, valor in dados:
            if chave and isinstance(chave, str):
                #salva/incrementa os tokens
                if chave in df.index:
                    df.at[chave, coluna_valor] += valor
                else:
                    df.loc[chave, coluna_valor] = valor

        #cria a coluna chave e ordena por valor do maior para o menor
        df.index.name = 'chave'
        df_ordenado = df.sort_values(by=df.columns[0], ascending=False)
        df_ordenado.to_csv(str(self.__lista_tokens))

    def __unir_dicionarios(self, dict_adicional:dict, dict_saida:dict)->dict:
        """
        Método acessório que une o dicionario adicional ao dicionário de saída
            Params:
                dict_adicional:dict = dicionário adicional que será unido
                dict_saida:dict = dicionário de saída que recebe os dados
            Return:
                dict = valor do dicionário de saída
        """        
        for chave_1 in dict_adicional.keys():
            if chave_1 in dict_saida.keys():
                dict_saida[chave_1] += dict_adicional[chave_1]
            else:
                dict_saida[chave_1] = dict_adicional[chave_1]
        del dict_adicional
        return dict_saida

    def processar_texto(self, path:Path, quantidade=150000):
        '''
        Método principal do processador de texto. Percorre a quantidade de vezes para fazer o BPE
            Params:
                path:Path = Caminho até o arquivo
                quantidade: int = número de iterações para gerar os tokens. Padrão 150000
        '''
        dict_char = self.__contar_caracteres(path)
        coringa = self.__achar_caractere_coringa(dict_char)

        #carregando o csv
        df = self.__carregar_tokens_csv()
        df = df.reset_index()
        lista_bpe = defaultdict(int, df.values.tolist())

        pos_index, tk_set = self.__carregar_savepoint()
        if pos_index == -1:
            lista_bpe = self.__unir_dicionarios(dict_char, lista_bpe)
        
        # faz um loop até o total de iterações solicitado +1 poi começa de 1, não 0
        for i in range(1,quantidade+1):
            if i> pos_index:
                #carrega o csv e ordena
                chaves_ordenadas = sorted(lista_bpe, key=lista_bpe.get, reverse=True)
                # se não foi processado aplica o BPE
                for char_chave in chaves_ordenadas:
                    if char_chave not in tk_set:
                        percentil = quantidade/100.0          
                        if int(i%percentil) ==0:
                            print(f"\t>>> Processado {(i/quantidade)*100:.2f}% do texto {path.name}: {i} de {quantidade}")
                            self.__salvar_tokens_csv(list(lista_bpe.items()))
                            self.__marcar_savepoint(i, tk_set)
                        lista_bpe = self.__aplicar_bpe(lista_bpe, path, char_chave, coringa)
                        tk_set.add(char_chave)
                        break
        
        #salva os dados como csv
        self.__marcar_savepoint(-1,set())
        self.__salvar_tokens_csv(list(lista_bpe.items()))
        tk_set = set()
    
    def __marcar_savepoint(self, pos, setlist):
        try:
            df = pd.read_csv(str(self.__savepoint),index_col=0)
        except Exception as e:
            df = pd.DataFrame(columns=['valor'])
        coluna_valor = df.columns[0]
        df.index.name = 'chave'
        #itera os dados para salvar os valores
        df.loc['setlist', coluna_valor] = str(set(setlist))
        df.loc['pos', coluna_valor] = str(pos)
        df.to_csv(str(self.__savepoint))
    
    def __carregar_savepoint(self)-> tuple[int, set]:
        try:
            df = pd.read_csv(str(self.__savepoint),index_col=0)
            coluna_valor = df.columns[0]
            pos_str = df.loc['pos', coluna_valor]
            setlist_str = df.loc['setlist', coluna_valor]
            return int(pos_str), ast.literal_eval(setlist_str)
        except FileNotFoundError as e:
            return -1, set()

    def get_tokens(self):
        df = self.__carregar_tokens_csv()
        df.insert(0, 'id', range(1, len(df) + 1))
        return df.sort_values('valor', ascending=False)