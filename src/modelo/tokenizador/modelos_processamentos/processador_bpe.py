from pathlib import Path
import itertools
from collections import Counter
from collections import defaultdict
from operator import itemgetter
import pandas as pd
import json

from src.ferramentas.ferramentas import texto_para_hex, hex_para_texto

class Processador_BPE:
    def __init__(self):
        self.__lista_tokens = Path('src/media/dados_processados/tokens.csv')

    def __contar_caracteres_texto(self, path:Path):
        lista_bpe = defaultdict()
        with open(str(path), encoding='utf-8') as f:
            texto = f.read()
            for i in texto:
                try:
                    lista_bpe[i]+=1
                except KeyError:
                    lista_bpe[i]=1
        caracteres = list(lista_bpe.items())
        self.__salvar_csv(caracteres)
        return lista_bpe

    def __achar_caractere_coringa(self, caracteres:list)->str:        
        for codigo in range(32, int(0x10FFFF), 1):
            caractere = chr(codigo)
            if caractere not in caracteres.keys() and caractere not in [' ', '\n']:
                return caractere

    def __contador_bpe(self,path:Path, caractere_chave:str, coringa:str):       
        lista_bpe = defaultdict() 
        with open(str(path), encoding='utf-8') as f:
            texto = f.read()
            texto = texto.replace(caractere_chave, coringa)
            for i in range(len(texto)-1):
                if texto[i] == coringa:
                    try:
                        lista_bpe[caractere_chave+texto[i+1:i+2]] += 1
                    except KeyError:
                        lista_bpe[caractere_chave+texto[i+1:i+2]] = 1
        self.__salvar_csv(list(lista_bpe.items()))
        return lista_bpe
    
    def __abrir_csv(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(str(self.__lista_tokens),index_col=0)
        except Exception as e:
            df = pd.DataFrame(columns=['valor'])
        return df

    def __salvar_csv(self, dados:list):
        df = self.__abrir_csv()
        coluna_valor = df.columns[0]
        for chave, valor in dados:
            if chave in df.index:
                # Incrementa o valor existente
                df.at[chave, coluna_valor] += valor
            else:
                df.loc[chave, coluna_valor] = valor

        #cria a coluna chave e ordena por valor do maior para o menor
        df.index.name = 'chave'
        df_ordenado = df.sort_values(by=df.columns[0], ascending=False)
        df_ordenado.to_csv(str(self.__lista_tokens))

    def processar_texto(self, path_texto:Path, quantidade=150000):
        caracteres = self.__contar_caracteres_texto(path_texto)
        coringa = self.__achar_caractere_coringa(caracteres)

        set_processados = set(self.__carregar_posicao()[1])
        pos =self.__carregar_posicao()[0]
        for i in range(1,quantidade+1):
            try:
                if i>pos:
                    df = self.__abrir_csv()
                    df = df.sort_values(by=df.columns[0], ascending=False)
                    df = df.reset_index()
                    for valor_livre in df.values.tolist():
                        char_chave = valor_livre[0]

                        if char_chave not in set_processados and isinstance(char_chave, str) and len(char_chave)>0:
                            self.__contador_bpe(path_texto, char_chave, coringa)
                            set_processados.add(char_chave)
            except KeyboardInterrupt:
                self.__salvar_posicao(i,set_processados)
                import gc
                gc.collect()


    def __salvar_posicao(self, loop:int, set_list:set ):
        dados = {'loop':loop, 'set_list':list(set_list)}
        with open('src/media/dados_processados/pos.json', 'w', encoding='utf-8') as f:
            json.dump(dados, f, indent=4, ensure_ascii=False)
    
    def __carregar_posicao(self)->list[int,list]:
        try:
            with open('src/media/dados_processados/pos.json', 'r', encoding='utf-8') as f:
                dados = json.load(f)
                return [dados['loop'], dados['set_list']]
        except Exception as e:
            return [-1,[]]