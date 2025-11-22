# Imports de terceros
from fastapi import Request
from starlette.types import ASGIApp, Receive, Scope, Send

# Imports locales
from database.database import SessionLocal
from model.models import Historial
from repository.historial_repository import crear_historial


class HistorialMiddleware:
    """
    Middleware para registrar todas las peticiones HTTP en el historial.
    
    Captura información de cada petición (método, endpoint, IP, user agent)
    y crea un registro en la base de datos con una descripción de la acción.
    
    Attributes:
        app: Aplicación ASGI a la que se aplica el middleware.
    """
    
    def __init__(self, app: ASGIApp):
        """
        Inicializa el middleware de historial.
        
        Args:
            app: Aplicación ASGI.
        """
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        """
        Procesa las peticiones HTTP y registra la acción en el historial.
        
        Args:
            scope: Información de la petición.
            receive: Canal de recepción de mensajes.
            send: Canal de envío de mensajes.
        """
        if scope["type"] == "http":
            request = Request(scope, receive=receive)
            ip = request.client.host
            user_agent = request.headers.get("user-agent")
            endpoint = scope["path"]
            metodo = scope["method"]

            db = SessionLocal()

            async def send_wrapper(message):
                """
                Wrapper para interceptar el inicio de la respuesta HTTP.
                
                Registra la acción en el historial antes de enviar la respuesta.
                
                Args:
                    message: Mensaje ASGI a enviar.
                """
                if message["type"] == "http.response.start":
                    # Determinar la acción según el endpoint y método
                    if endpoint == "/subirUsuario":
                        accion = "Creación de usuario"
                    elif endpoint == "/compararCara":
                        accion = "Intento de acceso via rostro"
                    elif endpoint.startswith("/usuarios") and metodo == "GET":
                        accion = "Consulta de usuario(s)"
                    elif endpoint.startswith("/usuarios") and metodo == "PUT":
                        accion = "Actualización de usuario"
                    elif endpoint.startswith("/usuarios") and metodo == "DELETE":
                        accion = "Eliminación de usuario"
                    elif endpoint == "/historial":
                        accion = "Consulta de historial"
                    else:
                        accion = f"Request a {endpoint}"

                    historial = Historial(
                        accion=accion,
                        metodo=metodo,
                        endpoint=endpoint,
                        ip=ip,
                        user_agent=user_agent
                    )
                    crear_historial(db=db, historial=historial)

                await send(message)

            await self.app(scope, receive, send_wrapper)
            db.close()
        else:
            await self.app(scope, receive, send)
