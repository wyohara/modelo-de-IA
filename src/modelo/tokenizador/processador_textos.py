from pathlib import Path
import pandas as pd
import os

from src.modelo.tokenizador.modelos_processamentos.processador_bpe import Processador_BPE


class ProcessadorTextos():
    def __init__(self, modelo='bag'):
        self.__dataset = Path('src/media/dataset/')        
        self.__lista_arquivos_processados = Path('src/media/dados_processados/arquivos_processados.csv')
        self.__modelo = modelo
        self.__processador = Processador_BPE()

    def __is_arquivo_processado(self, arquivo:Path, modelo_processamento:str)->bool:
        try:
            df = pd.read_csv(self.__lista_arquivos_processados)
            return ((df['caminho'] == str(arquivo)) & (df['modelo_processamento'] == modelo_processamento)).any()
        except (FileNotFoundError, pd.errors.EmptyDataError, KeyError):
            return False
    
    def __marcar_como_processado(self, arquivo:Path, modelo_processamento:str)->bool:
        try:
            df = pd.read_csv(self.__lista_arquivos_processados)
        except (FileNotFoundError, pd.errors.EmptyDataError, KeyError):
            df = pd.DataFrame()
        nova_linha = {'caminho':str(arquivo), 'modelo_processamento': modelo_processamento}
        novo_df = pd.concat([df, pd.DataFrame([nova_linha])], ignore_index=True)
        
        novo_df.to_csv(self.__lista_arquivos_processados, index=False)
        return True

    def processar_arquivos_dataset(self):
        arquivos_dataset = [f for f in self.__dataset.iterdir() if f.is_file()]
        resultado = []
        
        #montagem da lista de nomes de arquivos processados
        for path_arquivo in arquivos_dataset:
            if not self.__is_arquivo_processado(str(path_arquivo), self.__modelo):
                coringa = self.__processador.processar_texto(path_arquivo)
                #self.__marcar_como_processado(str(path_arquivo), self.__modelo)