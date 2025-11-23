# Imports estándar
import random


tokens_validos = set()


def generar_token() -> str:
    """
    Genera un token numérico aleatorio de 6 dígitos.
    
    Returns:
        str: Token generado.
    """
    token = str(random.randint(100000, 999999))
    tokens_validos.add(token)
    return token


def validar_token(token: str) -> bool:
    """
    Valida si un token existe en el conjunto de tokens válidos.
    
    Args:
        token: Token a validar.
    
    Returns:
        bool: True si el token es válido, False en caso contrario.
    """
    return token in tokens_validos


def eliminar_token(token: str) -> None:
    """
    Elimina un token del conjunto de tokens válidos.
    
    Args:
        token: Token a eliminar.
    """
    tokens_validos.discard(token)
