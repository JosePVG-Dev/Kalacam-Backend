# Imports estándar
import re
from io import BytesIO
from typing import List, Optional

# Imports de terceros
import numpy as np
from deepface import DeepFace
from fastapi import HTTPException
from PIL import Image
from scipy.spatial.distance import cosine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

# Imports locales
from model.models import Usuario
from repository.usuario_repository import crear_usuario, obtener_usuarios
from service.storage_service import (
    eliminar_imagen,
    obtener_extension_desde_content_type,
    subir_imagen
)

# Constantes
UMBRAL_SIMILITUD = 0.37  # Umbral para considerar rostros similares/duplicados


def validarRostro(contenido: bytes) -> List[float]:
    """
    Valida una imagen y genera su embedding facial utilizando DeepFace (Facenet).

    Parámetros:
        contenido (bytes): Contenido de la imagen en bytes (por ejemplo, de un archivo subido).

    Retorna:
        List[float]: Embedding del rostro como lista de floats.

    Excepciones:
        HTTPException 400: Si no se envió contenido, no se detecta rostro o embedding vacío.
        HTTPException 500: Si ocurre cualquier otro error al procesar la imagen.
    """
    if not contenido:
        raise HTTPException(status_code=400, detail="No se envió contenido de imagen")

    try:
        img = Image.open(BytesIO(contenido))
        resultado = DeepFace.represent(img_path=np.array(img), model_name="Facenet")

        if not resultado or "embedding" not in resultado[0]:
            raise HTTPException(status_code=400, detail="No se detectó ningún rostro")

        embedding = resultado[0]["embedding"]

        if len(embedding) == 0:
            raise HTTPException(status_code=400, detail="Rostro inválido o embedding vacío")
        
        return embedding

    except ValueError:
        raise HTTPException(status_code=400, detail="No se detectó ningún rostro")

    except Exception:
        raise HTTPException(status_code=500, detail="Error procesando la imagen")


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
            raise HTTPException(
                status_code=409,
                detail="Este rostro ya está registrado."
            )


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
        raise HTTPException(status_code=400, detail="El nombre no puede estar vacío")
    if not apellido or not apellido.strip():
        raise HTTPException(status_code=400, detail="El apellido no puede estar vacío")
    if len(nombre) > 100 or len(apellido) > 100:
        raise HTTPException(status_code=400, detail="El nombre o apellido es demasiado largo")

    if not email or not email.strip():
        raise HTTPException(status_code=400, detail="El email no puede estar vacío")
    
    patron_email = r"^[\w\.-]+@[\w\.-]+\.\w+$"
    if not re.match(patron_email, email):
        raise HTTPException(status_code=400, detail="El email no tiene un formato válido")

    if not embedding or not isinstance(embedding, list) or not all(isinstance(x, (float, int)) for x in embedding):
        raise HTTPException(status_code=400, detail="Embedding inválido")

    # Validar que el rostro no esté duplicado
    validarRostroDuplicado(db, embedding)

    # Subir imagen al volumen si existe
    ruta_imagen = None
    if imagen:
        try:
            extension = obtener_extension_desde_content_type(content_type or "image/jpeg")
            ruta_imagen = subir_imagen(imagen, extension)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error al subir imagen: {str(e)}")

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
            raise HTTPException(
                status_code=409,
                detail=f"El email {email} ya está registrado. Por favor, use otro email."
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Error al crear usuario: {str(e)}"
            )


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
        raise HTTPException(status_code=404, detail="No hay usuarios registrados")

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
