#!/usr/bin/env python3
"""
Script para eliminar todas las tablas de la base de datos en Railway.
Este script NO requiere confirmación (útil para automatización).

Uso en Railway:
    railway run python eliminar_tablas_railway.py

O con variables de entorno:
    DB_USER=... DB_PASSWORD=... python eliminar_tablas_railway.py
"""

# Imports estándar
import os
import sys

# Imports de terceros
from sqlalchemy import text, inspect

# Imports locales
from database.database import engine, Base
from model.models import Usuario, Historial


def eliminar_todas_las_tablas():
    """
    Elimina todas las tablas de la base de datos.
    """
    try:
        print("🔍 Conectando a la base de datos...")
        
        # Verificar conexión
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        
        print("✅ Conexión exitosa")
        
        # Listar tablas existentes
        inspector = inspect(engine)
        tablas_existentes = inspector.get_table_names()
        
        if not tablas_existentes:
            print("ℹ️  No hay tablas para eliminar")
            return True
        
        print(f"📋 Tablas encontradas: {', '.join(tablas_existentes)}")
        print()
        
        # Eliminar todas las tablas usando transacción
        with engine.begin() as conn:
            print("🗑️  Eliminando todas las tablas...")
            
            # Método 1: Usar SQLAlchemy metadata
            Base.metadata.drop_all(bind=engine, checkfirst=True)
            
            # Método 2: Eliminar manualmente por si acaso
            for tabla in reversed(tablas_existentes):
                print(f"   ✓ Eliminando tabla '{tabla}'...")
                conn.execute(text(f"DROP TABLE IF EXISTS `{tabla}`"))
        
        print()
        print("✅ Todas las tablas eliminadas correctamente")
        print("💡 Las tablas se recrearán automáticamente al iniciar la aplicación")
        return True
        
    except Exception as e:
        print(f"❌ Error al eliminar tablas: {e}")
        print(f"   Tipo de error: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("🗑️  ELIMINACIÓN DE TABLAS EN RAILWAY")
    print("=" * 60)
    print()
    
    # Verificar variables de entorno
    required_vars = ["DB_USER", "DB_PASSWORD", "DB_HOST", "DB_NAME"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Variables de entorno faltantes: {', '.join(missing_vars)}")
        print("💡 Asegúrate de configurar las variables en Railway o en tu archivo .env")
        sys.exit(1)
    
    print("✅ Variables de entorno configuradas")
    print(f"   DB_HOST: {os.getenv('DB_HOST')}")
    print(f"   DB_NAME: {os.getenv('DB_NAME')}")
    print(f"   DB_USER: {os.getenv('DB_USER')}")
    print()
    
    # Ejecutar eliminación
    if eliminar_todas_las_tablas():
        print()
        print("=" * 60)
        print("✅ Proceso completado exitosamente")
        print("=" * 60)
        sys.exit(0)
    else:
        print()
        print("=" * 60)
        print("❌ Proceso falló")
        print("=" * 60)
        sys.exit(1)

