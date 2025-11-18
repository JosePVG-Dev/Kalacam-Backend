from fastapi import Request
from starlette.types import ASGIApp, Receive, Scope, Send
from database.database import SessionLocal
from model.models import Historial
from repository.historial_repository import crear_historial

class HistorialMiddleware:
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] == "http":
            request = Request(scope, receive=receive)
            ip = request.client.host
            user_agent = request.headers.get("user-agent")
            endpoint = scope["path"]
            metodo = scope["method"]

            db = SessionLocal()

            async def send_wrapper(message):
                if message["type"] == "http.response.start":
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
