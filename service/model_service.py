# Imports estándar
import os
import shutil
from typing import Optional

# RetinaFace y DeepFace se importan de forma lazy solo cuando se necesitan
# para evitar problemas de compatibilidad al iniciar el servidor
_RetinaFace = None
_DeepFace = None


def _descargar_modelo_desde_drive(url_drive: str, destino: str) -> bool:
    """
    Descarga un modelo desde Google Drive usando gdown.
    
    Args:
        url_drive: URL de Google Drive en formato 'https://drive.google.com/uc?id=FILE_ID'
        destino: Ruta donde guardar el archivo descargado
    
    Returns:
        bool: True si la descarga fue exitosa, False en caso contrario
    """
    try:
        import gdown
        
        url_original = url_drive
        
        # Convertir URL de Google Drive al formato correcto si es necesario
        if "drive.google.com/file/d/" in url_drive:
            # Extraer el ID del archivo
            file_id = url_drive.split("/file/d/")[1].split("/")[0]
            url_drive = f"https://drive.google.com/uc?id={file_id}"
            print(f"   🔄 URL convertida: {url_drive}")
        elif "id=" not in url_drive and "drive.google.com" in url_drive:
            # Si es un enlace compartido, extraer el ID
            if "/d/" in url_drive:
                file_id = url_drive.split("/d/")[1].split("/")[0]
                url_drive = f"https://drive.google.com/uc?id={file_id}"
                print(f"   🔄 URL convertida: {url_drive}")
        
        print(f"📥 Descargando modelo desde Google Drive...")
        print(f"   Destino: {destino}")
        
        # Asegurar que el directorio destino existe
        os.makedirs(os.path.dirname(destino), exist_ok=True)
        
        # Descargar con gdown
        gdown.download(url_drive, destino, quiet=False)
        
        if os.path.exists(destino):
            tamaño = os.path.getsize(destino) / (1024 * 1024)  # Tamaño en MB
            print(f"✅ Modelo descargado exitosamente: {destino} ({tamaño:.2f} MB)")
            return True
        else:
            print(f"❌ Error: El archivo no se descargó correctamente")
            return False
            
    except ImportError:
        print(f"❌ Error: gdown no está instalado. Instala con: pip install gdown")
        return False
    except Exception as e:
        print(f"❌ Error al descargar modelo desde Drive: {str(e)}")
        print(f"   Tipo de error: {type(e).__name__}")
        return False


def _copiar_modelos_locales_a_volumen(modelos_locales: str, modelos_weights: str) -> None:
    """
    Copia modelos desde la carpeta local del proyecto al volumen persistente.
    
    Args:
        modelos_locales: Ruta de la carpeta local con modelos (models/weights)
        modelos_weights: Ruta de destino en el volumen (/data/models/deepface/weights)
    """
    if not os.path.exists(modelos_locales):
        return
    
    for archivo in os.listdir(modelos_locales):
        if archivo.endswith(('.h5', '.pth')) or os.path.isdir(os.path.join(modelos_locales, archivo)):
            origen = os.path.join(modelos_locales, archivo)
            destino = os.path.join(modelos_weights, archivo)
            
            # Si es un directorio (como retinaface), copiar recursivamente
            if os.path.isdir(origen):
                if not os.path.exists(destino):
                    shutil.copytree(origen, destino)
                    print(f"✅ Directorio de modelo copiado: {archivo}")
            # Si es un archivo y no existe en destino, copiarlo
            elif not os.path.exists(destino):
                shutil.copy2(origen, destino)
                print(f"✅ Modelo copiado: {archivo}")


def _verificar_y_descargar_arcface(modelos_weights: str) -> None:
    """
    Verifica si existe el modelo ArcFace y lo descarga desde Google Drive si es necesario.
    
    Args:
        modelos_weights: Ruta donde deben estar los modelos (/data/models/deepface/weights)
    """
    modelo_arcface = os.path.join(modelos_weights, "arcface_weights.h5")
    print(f"🔍 Verificando modelo ArcFace en: {modelo_arcface}")
    
    if os.path.exists(modelo_arcface):
        print(f"✅ Modelo ArcFace ya existe: {modelo_arcface}")
        return
    
    print("⚠️ Modelo ArcFace no encontrado")
    url_arcface = os.getenv("ARCFACE_DRIVE_URL")
    
    if url_arcface:
        print(f"📥 URL de Google Drive configurada, intentando descargar...")
        print(f"   URL: {url_arcface}")
        resultado = _descargar_modelo_desde_drive(url_arcface, modelo_arcface)
        if resultado:
            print(f"✅ Modelo descargado exitosamente")
        else:
            print(f"❌ Error al descargar el modelo")
    else:
        print("⚠️ ARCFACE_DRIVE_URL no está configurada. Configura esta variable para descargar automáticamente.")


def _configurar_deepface_home() -> str:
    """
    Configura el directorio de modelos de DeepFace y prepara el entorno.
    
    Returns:
        str: Ruta base configurada para DeepFace (DEEPFACE_HOME)
    """
    # Configurar directorio de modelos
    # Usar MODELS_PATH si está configurado, sino construir desde VOLUMEN_PATH
    modelos_base = os.getenv("MODELS_PATH")
    if not modelos_base:
        volumen_path = os.getenv("VOLUMEN_PATH", "uploads")
        modelos_base = os.path.join(volumen_path, "models", "deepface")
    
    modelos_weights = os.path.join(modelos_base, "weights")
    # Crear todos los directorios necesarios
    os.makedirs(modelos_weights, exist_ok=True)
    
    # Si hay modelos en la carpeta local del proyecto, copiarlos al volumen
    proyecto_base = os.path.dirname(os.path.dirname(__file__))
    modelos_locales = os.path.join(proyecto_base, "models", "weights")
    _copiar_modelos_locales_a_volumen(modelos_locales, modelos_weights)
    
    # Verificar si falta el modelo ArcFace y descargarlo desde Google Drive si hay URL configurada
    _verificar_y_descargar_arcface(modelos_weights)
    
    # Configurar variable de entorno para DeepFace
    os.environ["DEEPFACE_HOME"] = modelos_base
    
    return modelos_base


def get_retinaface():
    """
    Importa RetinaFace de forma lazy solo cuando se necesita.
    
    Returns:
        RetinaFace class o None si no está disponible
    """
    global _RetinaFace
    if _RetinaFace is None:
        try:
            from retinaface import RetinaFace
            _RetinaFace = RetinaFace
        except ImportError:
            _RetinaFace = None
    return _RetinaFace


def inicializar_modelos():
    """
    Inicializa los modelos en el startup del servidor.
    Verifica y descarga modelos si es necesario, sin importar DeepFace todavía.
    """
    print("🔧 Inicializando directorios y modelos...")
    # Solo configurar directorios y verificar/descargar modelos
    # Sin importar DeepFace (se importa lazy cuando se necesite)
    modelos_base = _configurar_deepface_home()
    print(f"📁 Directorio de modelos configurado: {modelos_base}")


def get_deepface():
    """
    Importa DeepFace de forma lazy solo cuando se necesita.
    Configura el entorno de modelos automáticamente en la primera llamada.
    
    Returns:
        DeepFace module
    """
    global _DeepFace
    if _DeepFace is None:
        from deepface import DeepFace
        
        # Configurar el entorno de modelos antes de usar DeepFace
        _configurar_deepface_home()
        
        _DeepFace = DeepFace
    return _DeepFace

