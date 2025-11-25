# Imports estándar
import os
import uuid
from typing import Optional

# Imports de terceros
from dotenv import load_dotenv
from fastapi import HTTPException

# Cargar variables de entorno
load_dotenv()

VOLUMEN_PATH = os.getenv("VOLUMEN_PATH", "uploads")
IMAGENES_PATH = os.path.join(VOLUMEN_PATH, "images", "usuarios")
# Crear directorios necesarios
os.makedirs(IMAGENES_PATH, exist_ok=True)


def subir_imagen(contenido: bytes, extension: str = "jpg") -> Optional[str]:
    """
    Guarda una imagen en el volumen y retorna la ruta relativa.
    
    Args:
        contenido: Contenido de la imagen en bytes.
        extension: Extensión del archivo.
    
    Returns:
        str: Ruta relativa de la imagen guardada.
    
    Raises:
        HTTPException: Si hay error al guardar la imagen.
    """
    try:
        nombre_archivo = f"{uuid.uuid4()}.{extension}"
        ruta_completa = os.path.join(IMAGENES_PATH, nombre_archivo)
        
        with open(ruta_completa, 'wb') as f:
            f.write(contenido)
        
        ruta_relativa = f"usuarios/{nombre_archivo}"
        return ruta_relativa
    
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error guardar")


def eliminar_imagen(ruta_relativa: str) -> bool:
    """
    Elimina una imagen del volumen.
    
    Args:
        ruta_relativa: Ruta relativa de la imagen.
    
    Returns:
        bool: True si se eliminó correctamente, False en caso contrario.
    """
    if not ruta_relativa:
        return False
    
    try:
        ruta_completa = os.path.join(VOLUMEN_PATH, "images", ruta_relativa)
        if os.path.exists(ruta_completa):
            os.remove(ruta_completa)
            return True
    except Exception:
        pass
    
    return False


def obtener_ruta_completa(ruta_relativa: str) -> Optional[str]:
    """
    Obtiene la ruta completa del archivo desde la ruta relativa.
    
    Args:
        ruta_relativa: Ruta relativa guardada en BD.
    
    Returns:
        Optional[str]: Ruta completa del archivo, None si no existe.
    """
    if not ruta_relativa:
        return None
    
    return os.path.join(VOLUMEN_PATH, "images", ruta_relativa)


def obtener_extension_desde_content_type(content_type: str) -> str:
    """
    Obtiene la extensión del archivo desde el content-type.
    
    Args:
        content_type: Content-Type de la imagen.
    
    Returns:
        str: Extensión del archivo.
    """
    if "jpeg" in content_type or "jpg" in content_type:
        return "jpg"
    elif "png" in content_type:
        return "png"
    else:
        return "jpg"

