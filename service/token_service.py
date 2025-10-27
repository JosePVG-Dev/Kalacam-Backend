import random


tokens_validos = set()

def generar_token():
   token =  str(random.randint(100000, 999999))
   tokens_validos.add(token)
   return token


def validar_token(token):
    return token in tokens_validos

def eliminar_token(token):
    tokens_validos.discard(token)

