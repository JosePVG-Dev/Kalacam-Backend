from sqlalchemy import Column, Integer, String, JSON, DateTime
from database import Base
from datetime import datetime


class Usuario(Base):
    __tablename__ = "usuarios"

    id = Column(Integer, primary_key=True, index=True)
    nombre = Column(String(100), nullable=False)
    apellido = Column(String(100), nullable=False)
    email = Column(String(255), unique=True, nullable=True)
    embedding = Column(JSON, nullable=False) 

    
class Historial(Base):
    __tablename__ = "historial"

    id = Column(Integer, primary_key=True, index=True)
    accion = Column(String(100))
    metodo = Column(String(10))
    endpoint = Column(String(100))
    ip = Column(String(100))
    user_agent = Column(String(255))
    fecha = Column(DateTime, default=datetime.now)