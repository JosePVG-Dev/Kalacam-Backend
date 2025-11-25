# Imports estándar
import asyncio
import base64
import json
import os

# Imports de terceros
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile, WebSocket, WebSocketDisconnect
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
from service.usuario_service import validarRostroDuplicado, validarRostroRapido
from service.storage_service import (
    eliminar_imagen,
    obtener_extension_desde_content_type,
    obtener_ruta_completa,
    subir_imagen
)
from service.token_service import generar_token, validar_token
from service.model_service import inicializar_modelos


# -------------------- CONFIG --------------------
app = FastAPI(
    title="API de Reconocimiento Facial IoT",
    description="API para gestión de usuarios con reconocimiento facial",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    """
    Evento de startup: inicializa base de datos y modelos.
    """
    # Crear tablas de base de datos
    Base.metadata.create_all(bind=engine)
    
    # Inicializar modelos (verificar y descargar si es necesario)
    print("🔧 Inicializando modelos de DeepFace...")
    inicializar_modelos()
    print("✅ Modelos inicializados correctamente")
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middlewares
app.add_middleware(HistorialMiddleware)
app.add_middleware(AuthMiddleware)

# Security
bearer_scheme = HTTPBearer()

def auth_required(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    """
    Valida el token de autenticación.
    
    Args:
        credentials: Credenciales HTTP Bearer con el token.
    
    Raises:
        HTTPException: Si el token es inválido.
    """
    token = credentials.credentials
    if not validar_token(token):
        raise HTTPException(status_code=401, detail="Token invalido")


# -------------------- DEPENDENCIA DB --------------------
def get_db():
    """
    Proporciona una sesión de base de datos.
    
    Yields:
        Session: Sesión de SQLAlchemy.
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
    Maneja las peticiones OPTIONS para CORS.
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
        dict: Mensaje de confirmación.
    
    Raises:
        HTTPException: Si la imagen no es válida o el rostro ya está registrado.
    """
    if imagen.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Archivo invalido")

    contenido = await imagen.read()
    embedding = face_service.validarRostro(contenido)
    usuario_guardado = face_service.crearUsuario(db, nombre, apellido, email, embedding, contenido, imagen.content_type)

    return {"ok": True, "mensaje": f"el usuario {nombre} {apellido}, ha sido creado exitosamente"}

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
        raise HTTPException(status_code=404, detail="No encontrado")
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
        raise HTTPException(status_code=404, detail="No encontrado")
    
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
            raise HTTPException(status_code=400, detail="Error imagen")
    
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
        raise HTTPException(status_code=404, detail="No encontrado")
    
    # Eliminar imagen del volumen si existe
    if usuario.imagen:
        eliminar_imagen(usuario.imagen)
    
    if eliminar_usuario(db, usuario_id):
        return {"ok": True, "mensaje": "Eliminado"}
    else:
        raise HTTPException(status_code=404, detail="No encontrado")

# -------------------- SERVIR IMÁGENES (PROTEGIDO) --------------------
@app.get("/imagenes/{ruta:path}")
def servir_imagen(
    ruta: str,
    auth: None = Depends(auth_required)
):
    """
    Sirve imágenes desde el volumen.
    
    Args:
        ruta: Ruta relativa de la imagen.
        auth: Dependencia de autenticación.
    
    Returns:
        FileResponse: Imagen solicitada.
    
    Raises:
        HTTPException: Si la imagen no existe.
    """
    ruta_completa = obtener_ruta_completa(ruta)
    if not ruta_completa or not os.path.exists(ruta_completa):
        raise HTTPException(status_code=404, detail="No encontrada")
    
    # Determinar content-type según extensión
    if ruta.endswith('.png'):
        media_type = 'image/png'
    elif ruta.endswith('.jpg') or ruta.endswith('.jpeg'):
        media_type = 'image/jpeg'
    else:
        media_type = 'image/jpeg'
    
    return FileResponse(ruta_completa, media_type=media_type)   

# -------------------- VALIDAR ROSTRO RÁPIDO (PÚBLICO - PRUEBA) --------------------
@app.post("/ws/validarRostro")  
async def validar_rostro_rapido_endpoint(
    imagen: UploadFile = File(...)
):
    """
    Valida rápidamente si hay un rostro en la imagen.
    
    Args:
        imagen: Imagen del rostro a validar (JPEG o PNG).
    
    Returns:
        dict: Resultado de la validación con indicador si hay rostro detectado.
    
    Raises:
        HTTPException: Si la imagen es inválida o hay error al procesarla.
    """
    if imagen.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Archivo invalido. Solo JPEG o PNG")

    contenido = await imagen.read()
    
    try:
        rostro_detectado = validarRostroRapido(contenido)
        
        if rostro_detectado:
            return {
                "ok": True,
                "mensaje": "Rostro detectado correctamente",
                "rostro_detectado": True
            }
        else:
            return {
                "ok": False,
                "mensaje": "No se detectó ningún rostro en la imagen",
                "rostro_detectado": False
            }
    
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar imagen: {str(e)}")


# -------------------- COMPARAR CARA (PÚBLICO) --------------------
@app.post("/compararCara")
async def comparar_cara(
    imagen: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Compara un rostro con los usuarios registrados.
    
    Args:
        imagen: Imagen del rostro a comparar (JPEG o PNG).
        db: Sesión de base de datos.
    
    Returns:
        Response: Respuesta con código HTTP explícito.
            - 200: Si el rostro es reconocido (con token)
            - 401: Si el rostro no es reconocido
    
    Raises:
        HTTPException: Si la imagen es inválida (400).
    """
    if imagen.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Archivo invalido")

    contenido = await imagen.read()
    nombre_usuario = face_service.compararRostro(db, contenido)

    if nombre_usuario:
        token = generar_token()
        return Response(
            status_code=200,
            content=json.dumps({"token": f"Hola {nombre_usuario}, token: {token}"}),
            media_type="application/json"
        )
    else:
        return Response(
            status_code=401,
            content=json.dumps({"mensaje": "No reconocido"}),
            media_type="application/json"
        )


# -------------------- HISTORIAL PROTEGIDO --------------------
@app.get("/historial")
def listar_historial(
    db: Session = Depends(get_db),
    auth: None = Depends(auth_required)
):
    """
    Lista el historial de acciones.
    
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
    Genera un token de prueba.
    
    Returns:
        dict: Token generado.
    """
    token = generar_token()
    return {"ok": True, "token": token}


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
        return {"ok": True}
    else:
        raise HTTPException(status_code=401, detail="Token invalido")


# -------------------- WEBSOCKET PRUEBA --------------------
@app.websocket("/ws/prueba")
async def websocket_prueba(websocket: WebSocket):
    """
    Endpoint WebSocket de prueba bidireccional.
    """
    await websocket.accept()
    print("Cliente conectado al WebSocket de prueba")
    
    await websocket.send_text("¡Conectado! Puedes enviarme mensajes y recibirás respuestas. También recibirás mensajes automáticos cada 5 segundos.")
    
    contador_automatico = 0
    contador_mensajes = 0
    
    try:
        while True:
            try:
                mensaje_cliente = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=5.0
                )
                
                contador_mensajes += 1
                print(f"Mensaje recibido del cliente: {mensaje_cliente}")
                
                respuesta = f"Servidor recibió tu mensaje #{contador_mensajes}: '{mensaje_cliente}'. Respuesta automática del servidor."
                await websocket.send_text(respuesta)
                
            except asyncio.TimeoutError:
                contador_automatico += 1
                mensaje_automatico = f"Mensaje automático #{contador_automatico} - Hora: {asyncio.get_event_loop().time():.2f}"
                await websocket.send_text(mensaje_automatico)
                print(f"Mensaje automático enviado: {mensaje_automatico}")
            
    except WebSocketDisconnect:
        print("Cliente desconectado del WebSocket de prueba")
    except Exception as e:
        print(f"Error en WebSocket: {e}")


@app.websocket("/ws/validarRostro")
async def websocket_validar_rostro(websocket: WebSocket):
    """
    Endpoint WebSocket que valida si hay un rostro en la imagen.
    
    Formato de mensaje esperado:
    - Como bytes (imagen directa)
    - Como JSON con base64: {"tipo": "imagen", "imagen": "base64_string", "content_type": "image/jpeg"}
    
    Respuesta:
    {"ok": true/false, "mensaje": "descripción", "rostro_detectado": true/false}
    """
    await websocket.accept()
    print("Cliente conectado al WebSocket de validarRostro")
    
    await websocket.send_text(json.dumps({
        "tipo": "conexion",
        "mensaje": "Conectado. Envía una imagen para validar el rostro."
    }))
    
    try:
        while True:
            try:
                mensaje = await asyncio.wait_for(
                    websocket.receive(),
                    timeout=30.0
                )
            except asyncio.TimeoutError:
                try:
                    await websocket.send_text(json.dumps({
                        "tipo": "keepalive",
                        "mensaje": "Conexión activa"
                    }))
                    continue
                except:
                    break
            
            contenido_imagen = None
            content_type = None
            
            if "bytes" in mensaje:
                contenido_imagen = mensaje["bytes"]
                content_type = "image/jpeg"
                print("Imagen recibida como bytes")
            
            elif "text" in mensaje:
                try:
                    data = json.loads(mensaje["text"])
                    
                    if data.get("tipo") == "keepalive":
                        continue
                    
                    if data.get("tipo") == "imagen":
                        imagen_base64 = data.get("imagen", "")
                        content_type = data.get("content_type", "image/jpeg")
                        contenido_imagen = base64.b64decode(imagen_base64)
                        print("Imagen recibida como base64")
                    else:
                        await websocket.send_text(json.dumps({
                            "ok": False,
                            "mensaje": "Formato JSON inválido"
                        }))
                        continue
                
                except json.JSONDecodeError:
                    await websocket.send_text(json.dumps({
                        "ok": False,
                        "mensaje": "Error al parsear JSON"
                    }))
                    continue
                except Exception as e:
                    await websocket.send_text(json.dumps({
                        "ok": False,
                        "mensaje": f"Error al procesar imagen: {str(e)}"
                    }))
                    continue
            
            if not contenido_imagen:
                await websocket.send_text(json.dumps({
                    "ok": False,
                    "mensaje": "No se recibió imagen válida"
                }))
                continue
            
            if content_type not in ["image/jpeg", "image/png"]:
                await websocket.send_text(json.dumps({
                    "ok": False,
                    "mensaje": "Tipo de archivo no permitido. Solo JPEG o PNG"
                }))
                continue
            
            try:
                print("Iniciando detección rápida de rostro...")
                
                rostro_detectado = validarRostroRapido(contenido_imagen)
                
                if rostro_detectado:
                    await websocket.send_text(json.dumps({
                        "ok": True,
                        "mensaje": "Rostro detectado correctamente",
                        "rostro_detectado": True
                    }))
                    print("Rostro detectado exitosamente")
                else:
                    await websocket.send_text(json.dumps({
                        "ok": False,
                        "mensaje": "No se detectó ningún rostro en la imagen",
                        "rostro_detectado": False
                    }))
                    print("No se detectó rostro")
                
            except HTTPException as e:
                await websocket.send_text(json.dumps({
                    "ok": False,
                    "mensaje": e.detail,
                    "status_code": e.status_code
                }))
                print(f"Error de validación: {e.detail}")
            
            except Exception as e:
                await websocket.send_text(json.dumps({
                    "ok": False,
                    "mensaje": f"Error al procesar imagen: {str(e)}"
                }))
                print(f"Error inesperado: {str(e)}")
    
    except WebSocketDisconnect:
        print("Cliente desconectado del WebSocket de validarRostro")
    except Exception as e:
        print(f"Error en WebSocket validarRostro: {e}")
