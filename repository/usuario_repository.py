# Imports de terceros
from sqlalchemy.orm import Session

# Imports locales
from model.models import Usuario


def crear_usuario(db: Session, usuario: Usuario) -> Usuario:
    """
    Crea un nuevo usuario en la base de datos.
    
    Args:
        db: Sesión de SQLAlchemy.
        usuario: Instancia del modelo Usuario.
    
    Returns:
        Usuario: Usuario creado con su ID asignado.
    """
    db.add(usuario)
    db.commit()
    db.refresh(usuario)
    return usuario


def obtener_usuarios(db: Session) -> list[Usuario]:
    """
    Obtiene todos los usuarios registrados.
    
    Args:
        db: Sesión de SQLAlchemy.
    
    Returns:
        list[Usuario]: Lista de todos los usuarios.
    """
    return db.query(Usuario).all()


def obtener_usuario(db: Session, usuario_id: int) -> Usuario | None:
    """
    Obtiene un usuario por su ID.
    
    Args:
        db: Sesión de SQLAlchemy.
        usuario_id: ID del usuario a buscar.
    
    Returns:
        Usuario | None: Usuario encontrado o None si no existe.
    """
    return db.query(Usuario).filter(Usuario.id == usuario_id).first()


def actualizar_usuario(db: Session, usuario_id: int, datos: dict) -> Usuario | None:
    """
    Actualiza los datos de un usuario existente.
    
    Args:
        db: Sesión de SQLAlchemy.
        usuario_id: ID del usuario a actualizar.
        datos: Diccionario con los campos a actualizar.
    
    Returns:
        Usuario | None: Usuario actualizado o None si no existe.
    """
    usuario = db.query(Usuario).filter(Usuario.id == usuario_id).first()
    if usuario:
        for key, value in datos.items():
            setattr(usuario, key, value)
        db.commit()
        db.refresh(usuario)
    return usuario


def eliminar_usuario(db: Session, usuario_id: int) -> bool:
    """
    Elimina un usuario de la base de datos.
    
    Args:
        db: Sesión de SQLAlchemy.
        usuario_id: ID del usuario a eliminar.
    
    Returns:
        bool: True si se eliminó exitosamente, False si no se encontró.
    """
    usuario = db.query(Usuario).filter(Usuario.id == usuario_id).first()
    if usuario:
        db.delete(usuario)
        db.commit()
        return True
    return False
