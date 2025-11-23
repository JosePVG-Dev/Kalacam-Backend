# Imports de terceros
from sqlalchemy.orm import Session

# Imports locales
from model.models import Historial


def crear_historial(db: Session, historial: Historial) -> Historial:
    """
    Crea un nuevo registro en el historial.
    
    Args:
        db: Sesión de SQLAlchemy.
        historial: Instancia del modelo Historial.
    
    Returns:
        Historial: Registro de historial creado con su ID asignado.
    """
    db.add(historial)
    db.commit()
    db.refresh(historial)
    return historial


def obtener_historial(db: Session) -> list[Historial]:
    """
    Obtiene todos los registros del historial ordenados por fecha descendente.
    
    Args:
        db: Sesión de SQLAlchemy.
    
    Returns:
        list[Historial]: Lista de registros del historial.
    """
    return db.query(Historial).order_by(Historial.fecha.desc()).all()
