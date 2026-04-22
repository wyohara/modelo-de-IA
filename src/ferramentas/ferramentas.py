
def texto_para_hex(texto:str):
    if type(texto)== str:
        return texto.encode('utf-8').hex()
    else:
        raise TypeError

def hex_para_texto(texto_hex):
    return bytes.fromhex(texto_hex.replace("#",'')).decode('utf-8',errors='surrogateescape')
