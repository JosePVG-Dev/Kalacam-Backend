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


def _verificar_y_descargar_arcface(modelos_base: str) -> None:
    """
    Verifica si existe el modelo ArcFace y lo descarga desde Google Drive si es necesario.
    
    DeepFace busca modelos en: {DEEPFACE_HOME}/.deepface/weights/
    Por eso descargamos directamente ahí.
    
    Args:
        modelos_base: Ruta base de modelos (DEEPFACE_HOME) (/data/models/deepface)
    """
    # DeepFace guarda modelos en .deepface/weights dentro de DEEPFACE_HOME
    deepface_weights = os.path.join(modelos_base, ".deepface", "weights")
    os.makedirs(deepface_weights, exist_ok=True)
    
    # Posibles nombres que DeepFace puede buscar para ArcFace
    posibles_nombres = [
        "arcface_weights.h5",
        "ArcFace.h5",
        "arcface.h5",
        "ArcFace_weights.h5"
    ]
    
    # Verificar si alguno de los posibles nombres ya existe
    modelo_existente = None
    for nombre in posibles_nombres:
        ruta = os.path.join(deepface_weights, nombre)
        if os.path.exists(ruta):
            modelo_existente = ruta
            print(f"✅ Modelo ArcFace encontrado: {ruta}")
            break
    
    if modelo_existente:
        return
    
    print(f"⚠️ Modelo ArcFace no encontrado en: {deepface_weights}")
    print(f"   Buscando nombres: {', '.join(posibles_nombres)}")
    
    # Listar archivos existentes para debugging
    if os.path.exists(deepface_weights):
        archivos = os.listdir(deepface_weights)
        if archivos:
            print(f"   📁 Archivos encontrados en weights/: {', '.join(archivos)}")
        else:
            print(f"   📁 Carpeta weights/ está vacía")
    
    # Obtener URL de descarga (personalizada o oficial de DeepFace)
    url_arcface = os.getenv("ARCFACE_DRIVE_URL")
    
    if not url_arcface:
        # URL oficial de DeepFace (extraída del código fuente de DeepFace)
        url_arcface = "https://drive.google.com/uc?id=1LVB3CdVejpmGHM28BpqqkbZP5hDEcdZY"
        print(f"⚠️ ARCFACE_DRIVE_URL no configurada, usando URL oficial de DeepFace")
    
    print(f"📥 Descargando modelo ArcFace...")
    print(f"   URL: {url_arcface}")
    
    # Descargar con el primer nombre (el más común)
    modelo_arcface = os.path.join(deepface_weights, posibles_nombres[0])
    resultado = _descargar_modelo_desde_drive(url_arcface, modelo_arcface)
    if resultado:
        print(f"✅ Modelo descargado exitosamente: {modelo_arcface}")
    else:
        print(f"❌ Error al descargar el modelo")
        print(f"   DeepFace intentará descargarlo automáticamente cuando se use por primera vez")


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
    
    # DeepFace guarda modelos en .deepface/weights dentro de DEEPFACE_HOME
    deepface_weights = os.path.join(modelos_base, ".deepface", "weights")
    os.makedirs(deepface_weights, exist_ok=True)
    
    # Si hay modelos en la carpeta local del proyecto, copiarlos al volumen
    proyecto_base = os.path.dirname(os.path.dirname(__file__))
    modelos_locales = os.path.join(proyecto_base, "models", "weights")
    if os.path.exists(modelos_locales):
        # Copiar modelos locales a la ubicación que DeepFace espera
        for archivo in os.listdir(modelos_locales):
            if archivo.endswith(('.h5', '.pth')) or os.path.isdir(os.path.join(modelos_locales, archivo)):
                origen = os.path.join(modelos_locales, archivo)
                destino = os.path.join(deepface_weights, archivo)
                if os.path.isdir(origen):
                    if not os.path.exists(destino):
                        shutil.copytree(origen, destino)
                        print(f"✅ Directorio de modelo copiado: {archivo}")
                elif not os.path.exists(destino):
                    shutil.copy2(origen, destino)
                    print(f"✅ Modelo copiado: {archivo}")
    
    # Verificar si falta el modelo ArcFace y descargarlo desde Google Drive si hay URL configurada
    # IMPORTANTE: Descargar directamente en la ubicación que DeepFace espera
    _verificar_y_descargar_arcface(modelos_base)
    
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


def _patch_deepface_url():
    """
    Monkey-patch para reemplazar la URL de descarga hardcodeada en DeepFace ArcFace.
    El link original de DeepFace está caído, usamos nuestra URL personalizada.
    """
    try:
        # Importar el módulo de ArcFace
        from deepface.basemodels import ArcFace
        import inspect
        
        # Obtener el código fuente de la función loadModel
        source = inspect.getsource(ArcFace.loadModel)
        
        # Si contiene la URL original, hacer monkey-patch
        if "https://drive.google.com/uc?id=1LVB3CdVejpmGHM28BpqqkbZP5hDEcdZY" in source:
            print("🔧 Aplicando monkey-patch a DeepFace ArcFace URL...")
            
            # Reemplazar la función loadModel con una versión parcheada
            original_loadModel = ArcFace.loadModel
            
            def patched_loadModel():
                # Guardar la función original de gdown
                import gdown
                original_download = gdown.download
                
                def patched_download(url, *args, **kwargs):
                    # Reemplazar URL si es la original (caída)
                    if url == "https://drive.google.com/uc?id=1LVB3CdVejpmGHM28BpqqkbZP5hDEcdZY":
                        custom_url = os.getenv("ARCFACE_DRIVE_URL", "https://drive.google.com/uc?id=1mjLC2mBJz71SDWnTcYTbrqE27RtOmMTk")
                        print(f"🔄 Reemplazando URL de DeepFace por: {custom_url}")
                        url = custom_url
                    return original_download(url, *args, **kwargs)
                
                # Aplicar el patch temporalmente
                gdown.download = patched_download
                try:
                    result = original_loadModel()
                finally:
                    # Restaurar la función original
                    gdown.download = original_download
                
                return result
            
            ArcFace.loadModel = patched_loadModel
            print("✅ Monkey-patch aplicado correctamente")
    except Exception as e:
        print(f"⚠️ No se pudo aplicar monkey-patch: {e}")
        print("   DeepFace usará su configuración por defecto")


def get_deepface():
    """
    Importa DeepFace de forma lazy solo cuando se necesita.
    Configura el entorno de modelos automáticamente en la primera llamada.
    
    IMPORTANTE: DEEPFACE_HOME debe configurarse ANTES de importar DeepFace,
    porque DeepFace lee esta variable al importarse.
    
    Returns:
        DeepFace module
    """
    global _DeepFace
    if _DeepFace is None:
        # CRÍTICO: Configurar DEEPFACE_HOME ANTES de importar DeepFace
        # DeepFace lee esta variable cuando se importa por primera vez
        modelos_base = _configurar_deepface_home()
        print(f"🔧 DEEPFACE_HOME configurado: {modelos_base}")
        print(f"🔧 Variable de entorno DEEPFACE_HOME: {os.getenv('DEEPFACE_HOME')}")
        
        # Verificar archivos antes de importar DeepFace
        deepface_weights = os.path.join(modelos_base, ".deepface", "weights")
        archivos_antes = []
        if os.path.exists(deepface_weights):
            archivos_antes = os.listdir(deepface_weights)
            print(f"🔧 Archivos en .deepface/weights/ ANTES de importar: {archivos_antes if archivos_antes else '(vacío)'}")
        
        # Ahora sí importar DeepFace (ya con DEEPFACE_HOME configurado)
        from deepface import DeepFace
        
        # Aplicar monkey-patch para usar nuestra URL personalizada
        _patch_deepface_url()
        
        # Verificar archivos después de importar (DeepFace puede haber creado/descargado archivos)
        if os.path.exists(deepface_weights):
            archivos_despues = os.listdir(deepface_weights)
            if archivos_despues != archivos_antes:
                nuevos = set(archivos_despues) - set(archivos_antes)
                if nuevos:
                    print(f"⚠️ DeepFace creó/descargó archivos después de importar: {list(nuevos)}")
                    print(f"   Esto significa que DeepFace no encontró tu modelo y descargó el suyo")
            print(f"🔧 Archivos en .deepface/weights/ DESPUÉS de importar: {archivos_despues if archivos_despues else '(vacío)'}")
        
        _DeepFace = DeepFace
    return _DeepFace

