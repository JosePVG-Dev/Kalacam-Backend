from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Form
from sqlalchemy.orm import Session
from database import SessionLocal
import numpy as np
from scipy.spatial.distance import cosine
from database import Base, engine
from models import Usuario

import face_service
from face_repository import obtener_usuarios

Base.metadata.create_all(bind=engine)

app = FastAPI()

# Dependencia para obtener la sesión de la base de datos
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.post("/subirUsuario")
async def subir_usuario(
    nombre: str = Form(...),
    apellido: str = Form(...),
    email: str = Form(...),
    imagen: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Sube una imagen, valida el rostro, genera embedding y crea un usuario en la DB.

    Parámetros:
        nombre (str): Nombre del usuario.
        apellido (str): Apellido del usuario.
        email (str): Correo electrónico del usuario.
        imagen (UploadFile): Imagen facial (JPEG o PNG).
        db (Session): Sesión activa de SQLAlchemy.

    Retorna:
        dict: Datos del usuario creado y longitud del embedding.
    """

    # Validar tipo de imagen
    if imagen.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Tipo de archivo no permitido")

    # Leer la imagen y obtener embedding
    contenido = await imagen.read()
    embedding = face_service.validarRostro(contenido)

    # Crear usuario (ahora con email)
    usuario_guardado = face_service.crearUsuario(db, nombre, apellido, email, embedding)

    return {
        "mensaje": "Usuario creado correctamente",
        "usuario": {
            "id": usuario_guardado.id,
            "nombre": usuario_guardado.nombre,
            "apellido": usuario_guardado.apellido,
            "email": usuario_guardado.email
        },
        "embedding_length": len(embedding)
    }


@app.post("/compararCara")
async def comparar_cara(imagen: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Compara una imagen con todos los usuarios guardados en la base de datos.
    Retorna el usuario más similar según la distancia coseno.
    """
    if imagen.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Tipo de archivo no permitido")

    contenido = await imagen.read()
    resultado = face_service.compararRostro(db, contenido)
    return resultado