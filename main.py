# Imports estándar
import os

# Imports de terceros
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

# Imports locales
from database.database import Base, engine, SessionLocal
from middleware.auth_middleware import AuthMiddleware
from middleware.historial_middleware import HistorialMiddleware
from model.models import Historial, TokenRequest, Usuario
from repository.historial_repository import crear_historial, obtener_historial
from repository.usuario_repository import (
    actualizar_usuario,
    eliminar_usuario,
    obtener_usuario,
    obtener_usuarios
)
import service.usuario_service as face_service
from service.usuario_service import validarRostroDuplicado
from service.storage_service import (
    eliminar_imagen,
    obtener_extension_desde_content_type,
    obtener_ruta_completa,
    subir_imagen
)
from service.token_service import generar_token, validar_token


# -------------------- CONFIG --------------------
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="API de Reconocimiento Facial IoT",
    description="API para gestión de usuarios con reconocimiento facial",
    version="1.0.0"
)
origins = [
    "http://localhost:3000",  # tu frontend
    "http://127.0.0.1:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # <- importante, acepta PUT y OPTIONS
    allow_headers=["*"],  # <- importante, acepta Authorization, Content-Type, etc.
)

# Middlewares
app.add_middleware(HistorialMiddleware)

# Security
bearer_scheme = HTTPBearer()

def auth_required(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    """
    Valida que el token de autenticación sea válido.
    
    Args:
        credentials: Credenciales HTTP Bearer con el token.
    
    Raises:
        HTTPException: Si el token es inválido.
    """
    token = credentials.credentials
    if not validar_token(token):
        raise HTTPException(status_code=401, detail="Token inválido")


# -------------------- DEPENDENCIA DB --------------------
def get_db():
    """
    Generador que proporciona una sesión de base de datos.
    
    Yields:
        Session: Sesión de SQLAlchemy para interactuar con la base de datos.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# -------------------- ENDPOINTS --------------------

# -------------------- OPTIONS HANDLER (CORS Preflight) --------------------
@app.options("/{full_path:path}")
async def options_handler(full_path: str, request: Request):
    """
    Maneja las peticiones OPTIONS (preflight) para CORS.
    Responde con los headers necesarios para permitir las peticiones cross-origin.
    """
    origin = request.headers.get("origin")
    
    # Verificar si el origen está permitido
    if origin in origins:
        return Response(
            status_code=200,
            headers={
                "Access-Control-Allow-Origin": origin,
                "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, PATCH, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type, Authorization, Accept, X-Requested-With",
                "Access-Control-Allow-Credentials": "true",
                "Access-Control-Max-Age": "3600",  # Cache por 1 hora
            }
        )
    else:
        # Origen no permitido
        return Response(status_code=403)

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
    Crea un nuevo usuario con reconocimiento facial.
    
    Args:
        nombre: Nombre del usuario.
        apellido: Apellido del usuario.
        email: Correo electrónico del usuario.
        imagen: Imagen del rostro del usuario (JPEG o PNG).
        db: Sesión de base de datos.
    
    Returns:
        dict: Mensaje de confirmación con el nombre del usuario creado.
    
    Raises:
        HTTPException: Si la imagen no es válida o el rostro ya está registrado.
    """
    if imagen.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Tipo de archivo no permitido")

    contenido = await imagen.read()
    embedding = face_service.validarRostro(contenido)
    usuario_guardado = face_service.crearUsuario(db, nombre, apellido, email, embedding, contenido, imagen.content_type)

    return {
        "mensaje": f"El usuario {nombre} {apellido}, ha sido creado exitosamente",
    }

# -------------------- USUARIOS PROTEGIDOS --------------------

@app.get("/usuarios")
def listar_usuarios(
    db: Session = Depends(get_db),
    auth: None = Depends(auth_required)
):
    """
    Lista todos los usuarios registrados.
    
    Args:
        db: Sesión de base de datos.
        auth: Dependencia de autenticación.
    
    Returns:
        list: Lista de todos los usuarios.
    """
    return obtener_usuarios(db)


@app.get("/usuarios/{usuario_id}")
def get_usuario(
    usuario_id: int,
    db: Session = Depends(get_db),
    auth: None = Depends(auth_required)
):
    """
    Obtiene un usuario específico por su ID.
    
    Args:
        usuario_id: ID del usuario a buscar.
        db: Sesión de base de datos.
        auth: Dependencia de autenticación.
    
    Returns:
        Usuario: Datos del usuario.
    
    Raises:
        HTTPException: Si el usuario no existe.
    """
    usuario = obtener_usuario(db, usuario_id)
    if not usuario:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    return usuario

@app.put("/usuarios/{usuario_id}")
async def update_usuario(
    usuario_id: int,
    nombre: str | None = Form(None),
    apellido: str | None = Form(None),
    email: str | None = Form(None),
    imagen: UploadFile | None = File(None),
    db: Session = Depends(get_db),
    auth: None = Depends(auth_required)
):
    """
    Actualiza los datos de un usuario existente.
    
    Args:
        usuario_id: ID del usuario a actualizar.
        nombre: Nuevo nombre (opcional).
        apellido: Nuevo apellido (opcional).
        email: Nuevo email (opcional).
        imagen: Nueva imagen del rostro (opcional).
        db: Sesión de base de datos.
        auth: Dependencia de autenticación.
    
    Returns:
        Usuario: Datos actualizados del usuario.
    
    Raises:
        HTTPException: Si el usuario no existe o hay error en la validación.
    """
    datos = {k: v for k, v in {"nombre": nombre, "apellido": apellido, "email": email}.items() if v is not None}
    
    # Procesar imagen si se proporciona
    if imagen:
        if imagen.content_type not in ["image/jpeg", "image/png"]:
            raise HTTPException(status_code=400, detail="Tipo de archivo no permitido")
    
    usuario_actualizado = actualizar_usuario(db, usuario_id, datos)
    if not usuario_actualizado:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    
    # Si hay imagen, procesar el nuevo rostro y actualizar
    if imagen:
        try:
            # Leer contenido de la imagen
            contenido = await imagen.read()
            
            # Validar rostro y generar nuevo embedding
            nuevo_embedding = face_service.validarRostro(contenido)
            
            # Validar que el nuevo rostro no esté duplicado (excluyendo al mismo usuario)
            validarRostroDuplicado(db, nuevo_embedding, excluir_usuario_id=usuario_id)
            
            # Eliminar imagen anterior si existe
            if usuario_actualizado.imagen:
                eliminar_imagen(usuario_actualizado.imagen)
            
            # Guardar nueva imagen
            extension = obtener_extension_desde_content_type(imagen.content_type)
            ruta_imagen = subir_imagen(contenido, extension)
            
            # Actualizar imagen Y embedding
            usuario_actualizado.imagen = ruta_imagen
            usuario_actualizado.embedding = nuevo_embedding
            
            db.commit()
            db.refresh(usuario_actualizado)
            
        except HTTPException:
            # Re-lanzar excepciones de validación (rostro duplicado, etc.)
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error al procesar la nueva imagen: {str(e)}")
    
    return usuario_actualizado

@app.delete("/usuarios/{usuario_id}")
def delete_usuario(
    usuario_id: int,
    db: Session = Depends(get_db),
    auth: None = Depends(auth_required)
):
    """
    Elimina un usuario y su imagen asociada.
    
    Args:
        usuario_id: ID del usuario a eliminar.
        db: Sesión de base de datos.
        auth: Dependencia de autenticación.
    
    Returns:
        dict: Mensaje de confirmación.
    
    Raises:
        HTTPException: Si el usuario no existe.
    """
    usuario = obtener_usuario(db, usuario_id)
    if not usuario:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    
    # Eliminar imagen del volumen si existe
    if usuario.imagen:
        eliminar_imagen(usuario.imagen)
    
    if eliminar_usuario(db, usuario_id):
        return {"mensaje": "Usuario eliminado correctamente"}
    else:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")

# -------------------- SERVIR IMÁGENES (PROTEGIDO) --------------------
@app.get("/imagenes/{ruta:path}")
def servir_imagen(
    ruta: str,
    auth: None = Depends(auth_required)
):
    """
    Sirve imágenes desde el volumen de Railway.
    Requiere autenticación con token.
    Ruta esperada: usuarios/uuid.jpg
    """
    ruta_completa = obtener_ruta_completa(ruta)
    if not ruta_completa or not os.path.exists(ruta_completa):
        raise HTTPException(status_code=404, detail="Imagen no encontrada")
    
    # Determinar content-type según extensión
    if ruta.endswith('.png'):
        media_type = 'image/png'
    elif ruta.endswith('.jpg') or ruta.endswith('.jpeg'):
        media_type = 'image/jpeg'
    else:
        media_type = 'image/jpeg'
    
    return FileResponse(ruta_completa, media_type=media_type)

# -------------------- COMPARAR CARA (PÚBLICO) --------------------
@app.post("/compararCara")
async def comparar_cara(
    imagen: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Compara un rostro con los usuarios registrados para reconocimiento facial.
    
    Args:
        imagen: Imagen del rostro a comparar (JPEG o PNG).
        db: Sesión de base de datos.
    
    Returns:
        dict: Token de autenticación si el rostro es reconocido.
    
    Raises:
        HTTPException: Si el rostro no es reconocido o la imagen es inválida.
    """
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
    """
    Lista el historial de acciones realizadas en el sistema.
    
    Args:
        db: Sesión de base de datos.
        auth: Dependencia de autenticación.
    
    Returns:
        list: Lista de registros del historial.
    """
    return obtener_historial(db)


# -------------------- GENERAR TOKEN (PRUEBAS) --------------------
@app.get("/generarToken")
def generar_token_prueba():
    """
    Endpoint de prueba para generar un token directamente.
    
    Útil para testing sin necesidad de pasar por el flujo de comparar cara.
    
    Returns:
        dict: Token generado y mensaje de confirmación.
    """
    token = generar_token()
    return {
        "token": token,
        "mensaje": "Token generado exitosamente para pruebas"
    }


# -------------------- LOGIN PÚBLICO --------------------
@app.post("/login")
def validar_token_endpoint(request: TokenRequest):
    """
    Valida si un token es válido para autenticación.
    
    Args:
        request: Objeto con el token a validar.
    
    Returns:
        dict: Confirmación si el token es válido.
    
    Raises:
        HTTPException: Si el token es inválido.
    """
    if validar_token(request.token):
        return {"ok": True, "mensaje": "Token válido, acceso permitido"}
    else:
        raise HTTPException(status_code=401, detail="Token inválido")
