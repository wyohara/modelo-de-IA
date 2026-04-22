from pathlib import Path
import pandas as pd
import os

from src.modelo.tokenizador.modelos_processamentos.processador_bpe import Processador_BPE


class ProcessadorTextos():
    '''
    Classe responsável por controlar a lista de textos do dataset e seu processamento
    '''
    def __init__(self, modelo='bag'):
        self.__dataset = Path('src/media/dataset/')        
        self.__lista_arquivos_processados = Path('src/media/dados_processados/arquivos_processados.csv')
        self.__modelo = modelo
        self.__processador = Processador_BPE()

    def __is_arquivo_processado(self, arquivo:Path, modelo_processamento:str)->bool:
        '''
        Método que verifica se o arquivo foi processado
            Params:
                arquivo:Path = caminho até o arquivo
                modelo_processamento:Path = nome do modelo usado
            Return:
                bool: True se foi processado e false se não encontrado ou erro
        '''
        try:
            df = pd.read_csv(self.__lista_arquivos_processados)
            return ((df['caminho'] == str(arquivo)) & (df['modelo_processamento'] == modelo_processamento)).any()
        except (FileNotFoundError, pd.errors.EmptyDataError, KeyError):
            return False
    
    def __marcar_como_processado(self, arquivo:Path, modelo_processamento:str)->bool:
        '''
        Método que marca o arquivo como processado e salva em um csv
            Params:
                arquivo:Path = caminho até o arquivo
                modelo_processamento:Path = nome do modelo usado
            Return:
                bool: True se foi processado e false se não encontrado ou erro
        '''
        #carregando o dataframe
        try:
            df = pd.read_csv(self.__lista_arquivos_processados)
        except (FileNotFoundError, pd.errors.EmptyDataError, KeyError):
            df = pd.DataFrame()
        
        #salvando a linha com o novo valor 
        nova_linha = {'caminho':str(arquivo), 'modelo_processamento': modelo_processamento}
        novo_df = pd.concat([df, pd.DataFrame([nova_linha])], ignore_index=True)        
        novo_df.to_csv(self.__lista_arquivos_processados, index=False)
        return True

    def processar_arquivos_dataset(self):
        '''
        Método principal que controla todo o processamento dos textos.
            - Carrega os caminhos dos texto
            - Se não foi processado aplica o processamento
        '''
        arquivos_dataset = [f for f in self.__dataset.iterdir() if f.is_file()]
        
        # percorre a lista de arquivos para processar
        for path_arquivo in arquivos_dataset:
            if not self.__is_arquivo_processado(str(path_arquivo), self.__modelo):
                # se não foi processado aplica o processamento e salva
                print(f">>> processando {path_arquivo.name}")
                self.__processador.processar_texto(path_arquivo)
                self.__marcar_como_processado(str(path_arquivo), self.__modelo)