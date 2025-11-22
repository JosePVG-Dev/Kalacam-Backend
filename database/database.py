# Imports estándar
import os

# Imports de terceros
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

# Cargar variables de entorno
load_dotenv()

# Configuración de base de datos desde variables de entorno
USER = os.getenv("DB_USER")
PASSWORD = os.getenv("DB_PASSWORD")
HOST = os.getenv("DB_HOST")
PORT = os.getenv("DB_PORT")
DBNAME = os.getenv("DB_NAME")

# URL de conexión a la base de datos MySQL
DATABASE_URL = f"mysql+pymysql://{USER}:{PASSWORD}@{HOST}:{PORT}/{DBNAME}"

# Motor de base de datos con NullPool (recomendado para transaction pooler en Railway)
engine = create_engine(DATABASE_URL, poolclass=NullPool)

# Generador de sesiones de base de datos
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Clase base para modelos declarativos
Base = declarative_base()
