# Imports estándar
import os
import re
from typing import List, Optional

# Imports de terceros
import cv2
import numpy as np
from fastapi import HTTPException
from scipy.spatial.distance import cosine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from model.models import Usuario
from repository.usuario_repository import crear_usuario, obtener_usuarios
from service.storage_service import (
    eliminar_imagen,
    obtener_extension_desde_content_type,
    subir_imagen
)

# Constantes
UMBRAL_SIMILITUD = 0.33  # Umbral para considerar rostros similares/duplicados

# Configuración del modelo de reconocimiento facial
# Modelos disponibles: VGG-Face, Facenet, Facenet512, OpenFace, DeepFace, DeepID, ArcFace, Dlib
# Detectores disponibles: retinaface, mtcnn, opencv, ssd, dlib, mediapipe, yolov8, yunet, fastmtcnn
MODELO_FACIAL = "Facenet512"  # Mejor precisión (99.41% LFW)
DETECTOR_BACKEND = "opencv"  # Detector rápido y ligero

# MediaPipe removido - usar DeepFace para detección

_deepface_module = None


def _get_deepface():
    """
    Obtiene el módulo DeepFace importado de forma lazy.
    """
    global _deepface_module
    if _deepface_module is None:
        from deepface import DeepFace as DeepFaceModule
        _deepface_module = DeepFaceModule
    return _deepface_module


def precargar_modelo_facial():
    """
    Pre-carga el modelo de reconocimiento facial al iniciar el servidor.
    Esto fuerza la descarga del modelo si no está disponible localmente.
    """
    try:
        DeepFace = _get_deepface()
        
        # Crear una imagen dummy válida (100x100 píxeles) para forzar la carga del modelo
        # Esto descargará el modelo si no existe, pero no procesará una imagen real
        # Usamos una imagen más grande para que sea válida para DeepFace
        img_dummy = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Intentar cargar el modelo haciendo una llamada mínima
        # Si el modelo no está descargado, DeepFace lo descargará automáticamente
        # Usamos enforce_detection=False para que no falle si no hay rostro
        DeepFace.represent(
            img_path=img_dummy,
            model_name=MODELO_FACIAL,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False
        )
        
    except Exception as e:
        # Si falla, intentar con build_model si está disponible
        try:
            DeepFace = _get_deepface()
            # Algunas versiones de DeepFace tienen build_model
            if hasattr(DeepFace, 'build_model'):
                DeepFace.build_model(MODELO_FACIAL)
        except Exception as e2:
            # No lanzamos excepción para que el servidor pueda arrancar igual
            # El modelo se descargará cuando se use por primera vez
            pass


def validarRostroRapido(contenido: bytes) -> bool:
    """
    Valida rápidamente si hay un rostro en la imagen (SIN generar embedding).
    Optimizado para WebSocket que necesita respuestas rápidas.
    
    Usa OpenCV CascadeClassifier para detección rápida de rostros (sin TensorFlow).
    Es más rápido que validarRostro() porque no genera embeddings.
    
    Parámetros:
        contenido (bytes): Contenido de la imagen en bytes.
    
    Retorna:
        bool: True si hay rostro detectado, False si no.
    
    Excepciones:
        HTTPException 400: Si no se envió contenido o imagen inválida.
    """
    if not contenido:
        raise HTTPException(status_code=400, detail="Sin imagen")
    
    try:
        # Convertir bytes a imagen numpy array
        nparr = np.frombuffer(contenido, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Imagen invalida")
        
        # Convertir a escala de grises para el detector
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Usar el detector de rostros de OpenCV (Haar Cascade)
        # Este detector viene incluido con OpenCV, no requiere TensorFlow
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(20, 20)
        )
        
        # Retornar True si se detectó al menos un rostro
        return len(faces) > 0
        
    except HTTPException:
        # Re-lanzar excepciones HTTP
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail="Error procesar imagen")


def validarRostro(contenido: bytes) -> List[float]:
    """
    Valida una imagen y genera su embedding facial utilizando DeepFace.

    Parámetros:
        contenido (bytes): Contenido de la imagen en bytes.

    Retorna:
        List[float]: Embedding facial como lista de números flotantes.

    Excepciones:
        HTTPException 400: Si no se envió contenido, imagen inválida o sin rostro detectado.
        HTTPException 500: Si ocurre un error al procesar la imagen.
    """
    if not contenido:
        raise HTTPException(status_code=400, detail="Sin imagen")

    try:
        nparr = np.frombuffer(contenido, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Imagen invalida")
        
        DeepFace = _get_deepface()
        resultado = DeepFace.represent(
            img_path=img,
            model_name=MODELO_FACIAL,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=True
        )

        # Normalizar resultado: DeepFace devuelve lista de dicts o un dict
        if isinstance(resultado, dict):
            resultado = [resultado]
        
        if not resultado or not isinstance(resultado[0], dict) or "embedding" not in resultado[0]:
            raise HTTPException(status_code=500, detail="Error inesperado")

        embedding = resultado[0]["embedding"]
        if not embedding:
            raise HTTPException(status_code=400, detail="Rostro invalido")

        return embedding

    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="Error procesar imagen")



def validarRostroDuplicado(db: Session, embedding: List[float], excluir_usuario_id: Optional[int] = None) -> None:
    """
    Valida que el embedding no corresponda a un rostro ya registrado.
    
    Parámetros:
        db (Session): Sesión activa de SQLAlchemy.
        embedding (List[float]): Embedding facial a validar.
        excluir_usuario_id (Optional[int]): ID del usuario a excluir de la validación (para actualizaciones).
    
    Excepciones:
        HTTPException 409: Si el rostro ya está registrado (similar a otro usuario).
    """
    # Convertir embedding a numpy array
    embedding_nuevo = np.array(embedding, dtype=float)
    
    # Obtener todos los usuarios registrados
    usuarios = obtener_usuarios(db)
    
    # Comparar con cada usuario existente
    for usuario in usuarios:
        # Excluir el usuario que se está actualizando
        if excluir_usuario_id and usuario.id == excluir_usuario_id:
            continue
            
        if not usuario.embedding:
            continue
        
        # Convertir embedding de BD a numpy array
        emb_db = np.array(usuario.embedding, dtype=float)
        
        # Calcular distancia de coseno
        distancia = cosine(embedding_nuevo, emb_db)
        
        # Si la distancia es menor al umbral, son rostros similares (duplicado)
        if distancia < UMBRAL_SIMILITUD:
            raise HTTPException(status_code=409, detail="Rostro duplicado")


def crearUsuario(db: Session, nombre: str, apellido: str, email: str, embedding: List[float], imagen: Optional[bytes] = None, content_type: Optional[str] = None) -> Usuario:
    """
    Crea un usuario en la base de datos después de validar sus datos.

    Parámetros:
        db (Session): Sesión activa de SQLAlchemy para interactuar con la base de datos.
        nombre (str): Nombre del usuario. No puede estar vacío ni superar 100 caracteres.
        apellido (str): Apellido del usuario. No puede estar vacío ni superar 100 caracteres.
        email (str): Correo electrónico del usuario. Debe tener un formato válido y no estar vacío.
        embedding (List[float]): Embedding facial generado previamente.
        imagen (Optional[bytes]): Contenido de la imagen en bytes. Se subirá al volumen.
        content_type (Optional[str]): Tipo de contenido de la imagen (ej: "image/jpeg").

    Retorna:
        Usuario: Objeto Usuario creado y guardado en la base de datos con su ID asignado.

    Excepciones:
        HTTPException 400: Si algún campo es inválido.
        HTTPException 409: Si el email o rostro ya está registrado.
    """

    if not nombre or not nombre.strip():
        raise HTTPException(status_code=400, detail="Nombre vacio")
    if not apellido or not apellido.strip():
        raise HTTPException(status_code=400, detail="Apellido vacio")
    if len(nombre) > 100 or len(apellido) > 100:
        raise HTTPException(status_code=400, detail="Nombre muy largo")

    if not email or not email.strip():
        raise HTTPException(status_code=400, detail="Email vacio")
    
    patron_email = r"^[\w\.-]+@[\w\.-]+\.\w+$"
    if not re.match(patron_email, email):
        raise HTTPException(status_code=400, detail="Email invalido")

    if not embedding or not isinstance(embedding, list) or not all(isinstance(x, (float, int)) for x in embedding):
        raise HTTPException(status_code=400, detail="Embedding invalido")

    # Validar que el rostro no esté duplicado
    validarRostroDuplicado(db, embedding)

    # Subir imagen al volumen si existe
    ruta_imagen = None
    if imagen:
        try:
            extension = obtener_extension_desde_content_type(content_type or "image/jpeg")
            ruta_imagen = subir_imagen(imagen, extension)
        except Exception as e:
            raise HTTPException(status_code=400, detail="Error subir")

    nuevo_usuario = Usuario(
        nombre=nombre.strip(),
        apellido=apellido.strip(),
        email=email.strip().lower(),
        embedding=embedding,
        imagen=ruta_imagen
    )

    try:
        usuario_guardado = crear_usuario(db, nuevo_usuario)
        return usuario_guardado
    except IntegrityError as e:
        db.rollback()
        # Verificar si es error de email duplicado
        if "email" in str(e.orig).lower() or "duplicate" in str(e.orig).lower():
            raise HTTPException(status_code=409, detail="Email duplicado")
        else:
            raise HTTPException(status_code=400, detail="Error crear")


def compararRostro(db: Session, contenido: bytes) -> Optional[str]:
    """
    Compara un rostro con los embeddings almacenados en la base de datos.
    
    Parámetros:
        db (Session): Sesión activa de SQLAlchemy.
        contenido (bytes): Contenido de la imagen a comparar.
    
    Retorna:
        Optional[str]: Nombre del usuario si el rostro fue reconocido, None si no.
    
    Excepciones:
        HTTPException 404: Si no hay usuarios registrados.
    """
    embedding_consulta = np.array(validarRostro(contenido))
    usuarios = obtener_usuarios(db)
    if not usuarios:
        raise HTTPException(status_code=404, detail="Sin usuarios")

    menor_distancia = float("inf")
    usuario_reconocido = None

    for usuario in usuarios:
        if not usuario.embedding:
            continue
        emb_db = np.array(usuario.embedding, dtype=float)
        distancia = cosine(embedding_consulta, emb_db)
        if distancia < menor_distancia:
            menor_distancia = distancia
            usuario_reconocido = usuario

    if menor_distancia < UMBRAL_SIMILITUD and usuario_reconocido:
        return usuario_reconocido.nombre
    
    return None