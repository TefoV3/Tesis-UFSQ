"""
Ejecuta este script UNA SOLA VEZ en Anaconda Prompt para autenticar TabPFN.
Después de hacerlo, el token queda guardado y el notebook nunca volverá a pedir login.

Uso:
    conda activate TesisCONDA
    python tabpfn_login.py
"""
import os, sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

print("=" * 60)
print("AUTENTICACIÓN DE TabPFN - solo necesitas hacer esto una vez")
print("=" * 60)
print()
print("Pasos:")
print("  1. Abre https://ux.priorlabs.ai/account en tu navegador")
print("  2. Regístrate o inicia sesión")
print("  3. Acepta la licencia en la pestaña 'Licenses'")
print("  4. Copia tu API Key")
print("  5. Pégala abajo cuando se pida")
print()

# TabPFN guarda el token en:
#   Windows: %APPDATA%\tabpfn\token  o  ~/.tabpfn/token
# Se puede pasar directamente como variable de entorno también.

api_key = input("Pega tu API Key aquí y presiona Enter: ").strip()
if not api_key:
    print("No pegaste ninguna clave. Sal con Ctrl+C e intenta de nuevo.")
    sys.exit(1)

# Guardar en la ubicación que TabPFN espera
import pathlib
token_dir = pathlib.Path.home() / ".tabpfn"
token_dir.mkdir(parents=True, exist_ok=True)
token_file = token_dir / "token"
token_file.write_text(api_key)
print(f"\nToken guardado en: {token_file}")

# Verificar que TabPFN lo lee bien
os.environ["TABPFN_TOKEN"] = api_key
try:
    import tabpfn
    print(f"tabpfn versión: {getattr(tabpfn,'__version__','?')}")
    # Verificación ligera: solo importar el clasificador (no hace fit todavía)
    from tabpfn import TabPFNClassifier
    print("\n✅ TabPFN autenticado correctamente. Ya puedes correr el notebook.")
except Exception as e:
    print(f"\n⚠️  Error al importar TabPFN: {e}")
    print("El token fue guardado igual. Intenta correr el notebook directamente.")
