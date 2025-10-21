from sqlalchemy.orm import Session
from models import Usuario  

def crear_usuario(db: Session, usuario: Usuario) -> Usuario:

    db.add(usuario)       # agrega al session
    db.commit()           
    db.refresh(usuario)   # refresca para obtener ID generado
    return usuario

def obtener_usuarios(db: Session) -> list[Usuario]:

    return db.query(Usuario).all()

