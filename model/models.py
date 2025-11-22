# Imports estándar
from datetime import datetime

# Imports de terceros
from pydantic import BaseModel
from sqlalchemy import Column, DateTime, Integer, JSON, String

# Imports locales
from database.database import Base


class Usuario(Base):
    """
    Modelo de datos para usuarios con reconocimiento facial.
    
    Attributes:
        id: Identificador único del usuario.
        nombre: Nombre del usuario (máximo 100 caracteres).
        apellido: Apellido del usuario (máximo 100 caracteres).
        email: Correo electrónico único del usuario (máximo 255 caracteres).
        embedding: Vector de embedding facial en formato JSON.
        imagen: Ruta relativa de la imagen en el volumen (ej: "usuarios/uuid.jpg").
    """
    __tablename__ = "usuarios"

    id = Column(Integer, primary_key=True, index=True)
    nombre = Column(String(100), nullable=False)
    apellido = Column(String(100), nullable=False)
    email = Column(String(255), unique=True, nullable=True)
    embedding = Column(JSON, nullable=False)
    imagen = Column(String(500), nullable=True)


class Historial(Base):
    """
    Modelo de datos para el historial de acciones del sistema.
    
    Attributes:
        id: Identificador único del registro.
        accion: Descripción de la acción realizada.
        metodo: Método HTTP usado (GET, POST, PUT, DELETE, etc.).
        endpoint: Endpoint de la API solicitado.
        ip: Dirección IP del cliente.
        user_agent: User agent del cliente.
        fecha: Fecha y hora de la acción.
    """
    __tablename__ = "historial"

    id = Column(Integer, primary_key=True, index=True)
    accion = Column(String(100))
    metodo = Column(String(10))
    endpoint = Column(String(100))
    ip = Column(String(100))
    user_agent = Column(String(255))
    fecha = Column(DateTime, default=datetime.now)


class TokenRequest(BaseModel):
    """
    Modelo Pydantic para solicitudes de validación de token.
    
    Attributes:
        token: Token a validar.
    """
    token: str
