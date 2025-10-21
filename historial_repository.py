
from sqlalchemy.orm import Session
from models import Historial
def crear_historial(db: Session, historial: Historial) -> Historial:

    db.add(historial)
    db.commit()
    db.refresh(historial)
    return historial

def obtener_historial(db: Session) -> list[Historial]:

    return db.query(Historial).order_by(Historial.fecha.desc()).all()