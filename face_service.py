from deepface import DeepFace
from PIL import Image
from io import BytesIO
import numpy as np
from fastapi import HTTPException
from scipy.spatial.distance import cosine
from typing import List
from sqlalchemy.orm import Session
from models import Usuario
from face_repository import crear_usuario, obtener_usuarios


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

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"No se detectó ningún rostro: {str(e)}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando la imagen: {str(e)}")


def crearUsuario(db: Session, nombre: str, apellido: str, embedding: List[float]) -> Usuario:
    """
    Crea un usuario en la base de datos después de validar sus datos.

    Parámetros:
        db (Session): Sesión activa de SQLAlchemy para interactuar con la base de datos.
        nombre (str): Nombre del usuario. No puede estar vacío ni superar 100 caracteres.
        apellido (str): Apellido del usuario. No puede estar vacío ni superar 100 caracteres.
        embedding (List[float]): Embedding facial generado previamente.

    Retorna:
        Usuario: Objeto Usuario creado y guardado en la base de datos con su ID asignado.

    Excepciones:
        HTTPException 400: Si algún campo es inválido (nombre/apellido vacío, embedding incorrecto).
    """
    if not nombre or not nombre.strip():
        raise HTTPException(status_code=400, detail="El nombre no puede estar vacío")
    if not apellido or not apellido.strip():
        raise HTTPException(status_code=400, detail="El apellido no puede estar vacío")
    if len(nombre) > 100 or len(apellido) > 100:
        raise HTTPException(status_code=400, detail="El nombre o apellido es demasiado largo")
    if not embedding or not isinstance(embedding, list) or not all(isinstance(x, (float, int)) for x in embedding):
        raise HTTPException(status_code=400, detail="Embedding inválido")

    nuevo_usuario = Usuario(
        nombre=nombre.strip(),
        apellido=apellido.strip(),
        embedding=embedding
    )
    usuario_guardado = crear_usuario(db, nuevo_usuario)
    return usuario_guardado


def compararRostro(db: Session, contenido: bytes):
    """
    Compara un rostro con los embeddings almacenados en la base de datos.
    Devuelve el usuario más parecido y la distancia coseno.
    """
    # 1️⃣ Generar embedding del rostro recibido
    embedding_consulta = np.array(validarRostro(contenido))

    # 2️⃣ Obtener usuarios de la base de datos
    usuarios = obtener_usuarios(db)
    if not usuarios:
        raise HTTPException(status_code=404, detail="No hay usuarios registrados")

    mejor_usuario = None
    menor_distancia = float("inf")

    # 3️⃣ Comparar con cada usuario
    for usuario in usuarios:
        if not usuario.embedding:
            continue

        emb_db = np.array(usuario.embedding, dtype=float)
        distancia = cosine(embedding_consulta, emb_db)

        if distancia < menor_distancia:
            menor_distancia = distancia
            mejor_usuario = usuario

    # 4️⃣ Evaluar similitud
    UMBRAL_SIMILITUD = 0.41
    if mejor_usuario and menor_distancia < UMBRAL_SIMILITUD:
        return {
            "mensaje": "Rostro reconocido",
            "usuario": {
                "id": mejor_usuario.id,
                "nombre": mejor_usuario.nombre,
                "apellido": mejor_usuario.apellido
            },
            "distancia": float(menor_distancia)
        }
    else:
        return {
            "mensaje": "No se encontró un rostro coincidente",
            "distancia_menor": float(menor_distancia)
        }