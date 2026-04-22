import pandas as pd
from pathlib import Path

class OperadorCSV:
    def __init__(self, arquivo: Path, headers=list):
        self.__arquivo = arquivo
        self.__headers = headers
    
    def carregar_arquivo(self, index=0)->pd.DataFrame:
        try:
            if index !=None:
                df = pd.read_csv(self.__arquivo, index_col=index)
            else:
                df = pd.read_csv(self.__arquivo)
            return df
        except (pd.errors.EmptyDataError,
                FileNotFoundError,
                PermissionError,
                UnicodeDecodeError,
                pd.errors.ParserError):
            return pd.DataFrame()

    def salvar_arquivo(self, df:pd.DataFrame)->bool:
        df.to_csv(self.__arquivo)
        return True
    
