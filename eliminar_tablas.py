from database.database import engine
from sqlalchemy import text

def eliminar_tablas():
    """
    Script temporal para eliminar las tablas de la base de datos
    """
    try:
        with engine.connect() as conn:
            # Eliminar tabla historial primero (por si hay dependencias)
            print("🗑️  Eliminando tabla 'historial'...")
            conn.execute(text("DROP TABLE IF EXISTS historial"))
            
            # Eliminar tabla usuarios
            print("🗑️  Eliminando tabla 'usuarios'...")
            conn.execute(text("DROP TABLE IF EXISTS usuarios"))
            
            conn.commit()
            
            print("✅ Tablas eliminadas correctamente")
            print("💡 Al reiniciar tu aplicación (main.py), las tablas se recrearán automáticamente")
            
    except Exception as e:
        print(f"❌ Error al eliminar tablas: {e}")

if __name__ == "__main__":
    print("⚠️  ADVERTENCIA: Este script eliminará las tablas 'usuarios' e 'historial'")
    print("⚠️  Todos los datos se perderán permanentemente")
    print()
    
    confirmar = input("¿Estás seguro? Escribe 'SI' para continuar: ")
    
    if confirmar.upper() == "SI":
        eliminar_tablas()
    else:
        print("❌ Operación cancelada")

