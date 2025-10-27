from sqlalchemy.orm import Session
from model.models import Usuario  

def crear_usuario(db: Session, usuario: Usuario) -> Usuario:
    db.add(usuario)
    db.commit()
    db.refresh(usuario)
    return usuario

def obtener_usuarios(db: Session) -> list[Usuario]:
    return db.query(Usuario).all()

def obtener_usuario(db: Session, usuario_id: int) -> Usuario | None:
    return db.query(Usuario).filter(Usuario.id == usuario_id).first()

def actualizar_usuario(db: Session, usuario_id: int, datos: dict) -> Usuario | None:
    usuario = db.query(Usuario).filter(Usuario.id == usuario_id).first()
    if usuario:
        for key, value in datos.items():
            setattr(usuario, key, value)
        db.commit()
        db.refresh(usuario)
    return usuario

def eliminar_usuario(db: Session, usuario_id: int) -> bool:
    usuario = db.query(Usuario).filter(Usuario.id == usuario_id).first()
    if usuario:
        db.delete(usuario)
        db.commit()
        return True
    return False
