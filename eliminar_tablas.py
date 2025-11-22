#!/usr/bin/env python3
"""
Script para eliminar todas las tablas de la base de datos.
Útil para resetear la base de datos en Railway o desarrollo local.

Uso:
    python eliminar_tablas.py
"""

# Imports estándar
import sys

# Imports de terceros
from sqlalchemy import text, inspect

# Imports locales
from database.database import engine, Base
from model.models import Usuario, Historial


def listar_tablas():
    """Lista todas las tablas existentes en la base de datos."""
    inspector = inspect(engine)
    tablas = inspector.get_table_names()
    return tablas


def eliminar_todas_las_tablas():
    """
    Elimina todas las tablas de la base de datos usando SQLAlchemy metadata.
    """
    try:
        print("🔍 Conectando a la base de datos...")
        
        # Listar tablas existentes
        tablas_existentes = listar_tablas()
        
        if not tablas_existentes:
            print("ℹ️  No hay tablas para eliminar")
            return True
        
        print(f"📋 Tablas encontradas: {', '.join(tablas_existentes)}")
        print()
        
        # Usar transacción para asegurar atomicidad
        with engine.begin() as conn:
            # Método 1: Usar SQLAlchemy metadata (recomendado)
            print("🗑️  Eliminando tablas usando SQLAlchemy metadata...")
            Base.metadata.drop_all(bind=engine, checkfirst=True)
            
            # Método 2: Eliminar manualmente por si acaso (backup)
            # Eliminar en orden inverso por si hay foreign keys
            for tabla in reversed(tablas_existentes):
                print(f"   ✓ Eliminando tabla '{tabla}'...")
                conn.execute(text(f"DROP TABLE IF EXISTS `{tabla}`"))
        
        print()
        print("✅ Todas las tablas eliminadas correctamente")
        print("💡 Al reiniciar tu aplicación (main.py), las tablas se recrearán automáticamente")
        return True
        
    except Exception as e:
        print(f"❌ Error al eliminar tablas: {e}")
        print(f"   Tipo de error: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False


def eliminar_tablas_especificas():
    """
    Elimina solo las tablas definidas en los modelos (usuarios e historial).
    """
    try:
        print("🔍 Conectando a la base de datos...")
        
        tablas_a_eliminar = ["historial", "usuarios"]
        
        with engine.begin() as conn:
            for tabla in tablas_a_eliminar:
                print(f"🗑️  Eliminando tabla '{tabla}'...")
                conn.execute(text(f"DROP TABLE IF EXISTS `{tabla}`"))
        
        print()
        print("✅ Tablas eliminadas correctamente")
        print("💡 Al reiniciar tu aplicación (main.py), las tablas se recrearán automáticamente")
        return True
        
    except Exception as e:
        print(f"❌ Error al eliminar tablas: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("⚠️  ADVERTENCIA: Este script eliminará las tablas de la base de datos")
    print("⚠️  Todos los datos se perderán permanentemente")
    print("=" * 60)
    print()
    
    # Verificar conexión primero
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("✅ Conexión a la base de datos exitosa")
    except Exception as e:
        print(f"❌ Error al conectar a la base de datos: {e}")
        print("💡 Verifica tus variables de entorno (DB_USER, DB_PASSWORD, DB_HOST, etc.)")
        sys.exit(1)
    
    print()
    print("Opciones:")
    print("1. Eliminar TODAS las tablas de la base de datos")
    print("2. Eliminar solo tablas de la aplicación (usuarios, historial)")
    print("3. Cancelar")
    print()
    
    opcion = input("Selecciona una opción (1/2/3): ").strip()
    
    if opcion == "1":
        print()
        confirmar = input("¿Estás SEGURO de eliminar TODAS las tablas? Escribe 'SI' para continuar: ")
        if confirmar.upper() == "SI":
            print()
            if eliminar_todas_las_tablas():
                sys.exit(0)
            else:
                sys.exit(1)
        else:
            print("❌ Operación cancelada")
            sys.exit(0)
    
    elif opcion == "2":
        print()
        confirmar = input("¿Estás seguro de eliminar las tablas 'usuarios' e 'historial'? Escribe 'SI' para continuar: ")
        if confirmar.upper() == "SI":
            print()
            if eliminar_tablas_especificas():
                sys.exit(0)
            else:
                sys.exit(1)
        else:
            print("❌ Operación cancelada")
            sys.exit(0)
    
    else:
        print("❌ Operación cancelada")
        sys.exit(0)

