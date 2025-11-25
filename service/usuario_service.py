# Imports estándar
import re
from io import BytesIO
from typing import List, Optional

# Imports de terceros
import cv2
import numpy as np
from fastapi import HTTPException
from PIL import Image
from scipy.spatial.distance import cosine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

# Imports locales
from service.model_service import get_deepface, get_retinaface
from model.models import Usuario
from repository.usuario_repository import crear_usuario, obtener_usuarios
from service.storage_service import (
    eliminar_imagen,
    obtener_extension_desde_content_type,
    subir_imagen
)

# Constantes
UMBRAL_SIMILITUD = 0.37  # Umbral para considerar rostros similares/duplicados

# Configuración del modelo de reconocimiento facial
# Modelos disponibles: VGG-Face, Facenet, Facenet512, OpenFace, DeepFace, DeepID, ArcFace, Dlib
# RetinaFace es el mejor detector, pero también puedes usar: mtcnn, opencv, ssd, dlib
MODELO_FACIAL = "ArcFace"  # Mejor precisión (99.41% LFW)
DETECTOR_BACKEND = "retinaface"  # Mejor detector de rostros

# MediaPipe removido - usar DeepFace para detección


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
            raise HTTPException(status_code=400, detail="Imagen inválida")
        
        # Convertir a escala de grises para el detector
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Usar el detector de rostros de OpenCV (Haar Cascade)
        # Este detector viene incluido con OpenCV, no requiere TensorFlow
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Retornar True si se detectó al menos un rostro
        return len(faces) > 0
        
    except HTTPException:
        # Re-lanzar excepciones HTTP
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al procesar imagen: {str(e)}")


def validarRostro(contenido: bytes) -> List[float]:
    """
    Valida una imagen y genera su embedding facial utilizando DeepFace.
    
    Modelo configurado: ArcFace (máxima precisión 99.41%)
    Detector: RetinaFace (mejor precisión en detección de rostros)

    Parámetros:
        contenido (bytes): Contenido de la imagen en bytes (por ejemplo, de un archivo subido).

    Retorna:
        List[float]: Embedding del rostro como lista de floats.

    Excepciones:
        HTTPException 400: Si no se envió contenido, no se detecta rostro o embedding vacío.
        HTTPException 500: Si ocurre cualquier otro error al procesar la imagen.
    """
    if not contenido:
        raise HTTPException(status_code=400, detail="Sin imagen")

    try:
        img = Image.open(BytesIO(contenido))
        DeepFace = get_deepface()
        resultado = DeepFace.represent(
            img_path=np.array(img),
            model_name=MODELO_FACIAL,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=True  # Fuerza la detección de rostro
        )

        if not resultado or "embedding" not in resultado[0]:
            raise HTTPException(status_code=400, detail="Sin rostro")

        embedding = resultado[0]["embedding"]

        if len(embedding) == 0:
            raise HTTPException(status_code=400, detail="Rostro invalido")
        
        return embedding

    except ValueError as e:
        raise HTTPException(status_code=400, detail="Sin rostro")
    
    except HTTPException:
        # Re-lanzar HTTPExceptions
        raise
    
    except ImportError as e:
        # Error al importar DeepFace o sus dependencias
        error_msg = str(e)
        print(f"Error de importación en validarRostro: {error_msg}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error de importación: DeepFace o sus dependencias no están disponibles. {error_msg}"
        )
    
    except Exception as e:
        # Log del error real para debugging
        error_msg = str(e)
        error_type = type(e).__name__
        print(f"Error en validarRostro: {error_type}: {error_msg}")
        
        # Retornar error más descriptivo con el tipo y mensaje
        raise HTTPException(
            status_code=500, 
            detail=f"Error al procesar imagen ({error_type}): {error_msg}"
        )


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
