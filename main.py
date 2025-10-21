from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Form, Request
from sqlalchemy.orm import Session
from database import SessionLocal
import numpy as np
from scipy.spatial.distance import cosine
from database import Base, engine
from models import Usuario

import face_service
from historial_repository import crear_historial

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
    request: Request,
    nombre: str = Form(...),
    apellido: str = Form(...),
    email: str = Form(...),
    imagen: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Sube una imagen, valida el rostro, genera embedding y crea un usuario en la DB.
    Además, registra la acción en la tabla de historial.
    """
    if imagen.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Tipo de archivo no permitido")

    contenido = await imagen.read()
    embedding = face_service.validarRostro(contenido)

    usuario_guardado = face_service.crearUsuario(db, nombre, apellido, email, embedding)

    ip = request.client.host
    user_agent = request.headers.get("user-agent")
    crear_historial(
        db=db,
        accion=f"Usuario {nombre} {apellido} creado",
        metodo="POST",
        endpoint="/subirUsuario",
        ip=ip,
        user_agent=user_agent
    )

    return {
        "mensaje": f"El usuario {nombre} {apellido}, ha sido creado exitosamente ",
    }


@app.post("/compararCara")
async def comparar_cara(
    request: Request,
    imagen: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Compara una imagen con todos los usuarios guardados en la base de datos.
    Registra el intento en la tabla de historial.
    """
    if imagen.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Tipo de archivo no permitido")

    contenido = await imagen.read()
    resultado = face_service.compararRostro(db, contenido)

    ip = request.client.host
    user_agent = request.headers.get("user-agent")

    mensaje = resultado.get("mensaje", "")
    if mensaje.startswith("Bienvenido"):
        accion = f"{mensaje} (acceso concedido)"
    else:
        accion = "Intento fallido de acceso (rostro no reconocido)"

    crear_historial(
        db=db,
        accion=accion,
        metodo="POST",
        endpoint="/compararCara",
        ip=ip,
        user_agent=user_agent
    )

    return resultado