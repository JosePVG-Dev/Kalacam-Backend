# Imports de terceros
from fastapi import HTTPException, Request
from starlette.types import ASGIApp, Receive, Scope, Send

# Imports locales
from service.token_service import tokens_validos


class AuthMiddleware:
    """
    Middleware para validar autenticación con tokens en todas las rutas.
    
    Verifica que las peticiones a endpoints protegidos incluyan un token
    válido en el header Authorization. Los endpoints públicos definidos
    en la lista son accesibles sin autenticación.
    
    Attributes:
        app: Aplicación ASGI a la que se aplica el middleware.
    """
    
    def __init__(self, app: ASGIApp):
        """
        Inicializa el middleware de autenticación.
        
        Args:
            app: Aplicación ASGI.
        """
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        """
        Procesa las peticiones HTTP validando autenticación.
        
        Args:
            scope: Información de la petición.
            receive: Canal de recepción de mensajes.
            send: Canal de envío de mensajes.
        
        Raises:
            HTTPException: Si la autenticación falla.
        """
        if scope["type"] == "http":
            request = Request(scope, receive=receive)
            path = scope["path"]

            # Endpoints que no requieren autenticación
            endpoints_publicos = ["/subirUsuario", "/login", "/docs", "/openapi.json", "/compararCara"]

            if path not in endpoints_publicos:
                auth_header = request.headers.get("Authorization")
                if not auth_header or not auth_header.startswith("Bearer "):
                    raise HTTPException(status_code=401, detail="Authorization header faltante o inválido")

                token = auth_header.split(" ")[1]

                if token not in tokens_validos:
                    raise HTTPException(status_code=401, detail="Token inválido")

            await self.app(scope, receive, send)
        else:
            await self.app(scope, receive, send)
