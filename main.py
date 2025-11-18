from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Form, Request, Header
from sqlalchemy.orm import Session
from database.database import SessionLocal
import numpy as np
from scipy.spatial.distance import cosine
from database.database import Base, engine
from model.models import Usuario, Historial, TokenRequest
from middleware.historial_middleware import HistorialMiddleware
from middleware.auth_middleware import AuthMiddleware
import service.usuario_service as face_service
from repository.historial_repository import crear_historial, obtener_historial
from service.token_service import validar_token, generar_token
from repository.usuario_repository import obtener_usuarios, obtener_usuario, actualizar_usuario, eliminar_usuario
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Depends, HTTPException

# -------------------- CONFIG --------------------
Base.metadata.create_all(bind=engine)
app = FastAPI()

# Middlewares
app.add_middleware(AuthMiddleware)
app.add_middleware(HistorialMiddleware)

# Security
bearer_scheme = HTTPBearer()

def auth_required(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    token = credentials.credentials
    if not validar_token(token):
        raise HTTPException(status_code=401, detail="Token inválido")

# -------------------- DEPENDENCIA DB --------------------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# -------------------- ENDPOINTS --------------------

@app.post("/subirUsuario")
async def subir_usuario(
    request: Request,
    nombre: str = Form(...),
    apellido: str = Form(...),
    email: str = Form(...),
    imagen: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    if imagen.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Tipo de archivo no permitido")

    contenido = await imagen.read()
    embedding = face_service.validarRostro(contenido)
    usuario_guardado = face_service.crearUsuario(db, nombre, apellido, email, embedding)

    return {
        "mensaje": f"El usuario {nombre} {apellido}, ha sido creado exitosamente",
    }

# -------------------- USUARIOS PROTEGIDOS --------------------

@app.get("/usuarios")
def listar_usuarios(
    db: Session = Depends(get_db),
    auth: None = Depends(auth_required)
):
    return obtener_usuarios(db)

@app.get("/usuarios/{usuario_id}")
def get_usuario(
    usuario_id: int,
    db: Session = Depends(get_db),
    auth: None = Depends(auth_required)
):
    usuario = obtener_usuario(db, usuario_id)
    if not usuario:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    return usuario

@app.put("/usuarios/{usuario_id}")
def update_usuario(
    usuario_id: int,
    nombre: str | None = None,
    apellido: str | None = None,
    email: str | None = None,
    db: Session = Depends(get_db),
    auth: None = Depends(auth_required)
):
    datos = {k: v for k, v in {"nombre": nombre, "apellido": apellido, "email": email}.items() if v is not None}
    usuario_actualizado = actualizar_usuario(db, usuario_id, datos)
    if not usuario_actualizado:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    return usuario_actualizado

@app.delete("/usuarios/{usuario_id}")
def delete_usuario(
    usuario_id: int,
    db: Session = Depends(get_db),
    auth: None = Depends(auth_required)
):
    if eliminar_usuario(db, usuario_id):
        return {"mensaje": "Usuario eliminado correctamente"}
    else:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")

# -------------------- COMPARAR CARA (PÚBLICO) --------------------
@app.post("/compararCara")
async def comparar_cara(
    imagen: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    if imagen.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Tipo de archivo no permitido")

    contenido = await imagen.read()
    nombre_usuario = face_service.compararRostro(db, contenido)

    if nombre_usuario:
        token = generar_token()
        return {"token": f"Hola {nombre_usuario}, token: {token}"}
    else:
        raise HTTPException(status_code=401, detail="Rostro no reconocido")

# -------------------- HISTORIAL PROTEGIDO --------------------
@app.get("/historial")
def listar_historial(
    db: Session = Depends(get_db),
    auth: None = Depends(auth_required)
):
    return obtener_historial(db)

# -------------------- LOGIN PÚBLICO --------------------
@app.post("/login")
def validar_token_endpoint(request: TokenRequest):
    if validar_token(request.token):
        return {"ok": True, "mensaje": "Token válido, acceso permitido"}
    else:
        raise HTTPException(status_code=401, detail="Token inválido")
