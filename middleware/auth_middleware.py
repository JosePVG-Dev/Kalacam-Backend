from fastapi import Request, HTTPException
from starlette.types import ASGIApp, Receive, Scope, Send
from service.token_service import tokens_validos 


class AuthMiddleware:
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] == "http":
            request = Request(scope, receive=receive)
            path = scope["path"]

            endpoints_publicos = ["/subirUsuario", "/login", "/docs", "/openapi.json","/compararCara"]

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
