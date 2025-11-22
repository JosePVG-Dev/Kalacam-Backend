# Imports estándar
import random


# NOTA: Esta implementación usa una variable global mutable para almacenar tokens en memoria.
# En producción, se recomienda usar una solución más robusta como:
# - Redis para gestión de sesiones
# - JWT (JSON Web Tokens) con firma criptográfica
# - Base de datos con tabla de sesiones/tokens
# Esta solución es adecuada solo para desarrollo/pruebas.
tokens_validos = set()


def generar_token() -> str:
    """
    Genera un token numérico aleatorio de 6 dígitos.
    
    El token se almacena en memoria para validación posterior.
    
    Returns:
        str: Token generado (número de 6 dígitos como string).
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
