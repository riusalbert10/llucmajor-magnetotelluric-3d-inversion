#!/usr/bin/env python
"""
Preparar archivos de entrada para inversión ModEM 2D
Para 67 estaciones .edi
"""
import mtpy.modeling.modem as modem
import numpy as np
import glob
import os

# ============ CONFIGURACIÓN ============
edi_path = r"C:\ruta\a\tus\edis"  # CAMBIAR ESTA RUTA
output_path = "./modem_input"
basename = "albert_inv2d"

# Parámetros de inversión
n_periods = 25  # Número de períodos/frecuencias
period_min = 0.001  # Período mínimo (segundos)
period_max = 1000   # Período máximo (segundos)
res_initial = 100   # Resistividad inicial (Ohm·m)

# Parámetros de malla 2D
cell_size_x = 500   # Tamaño de celda horizontal (metros)
cell_size_z = 500   # Tamaño de celda vertical (metros)
n_layers = 40       # Número de capas
z1_layer = 10       # Espesor primera capa (metros)

# Parámetros de suavizado
smoothing_x = 0.3
smoothing_z = 0.3

# =========================================

# Crear directorio de salida
os.makedirs(output_path, exist_ok=True)

print("="*60)
print("PREPARACIÓN DE INVERSIÓN ModEM 2D")
print("="*60)

# 1. Cargar archivos .edi
print(f"\n1. Buscando archivos .edi en: {edi_path}")
edi_files = sorted(glob.glob(os.path.join(edi_path, "*.edi")))
print(f"   Encontrados: {len(edi_files)} archivos")

if len(edi_files) == 0:
    print("   ❌ ERROR: No se encontraron archivos .edi")
    print(f"   Verifica la ruta: {edi_path}")
    exit(1)

# 2. Crear objeto Data de ModEM
print("\n2. Creando objeto Data de ModEM...")
data_obj = modem.Data(edi_list=edi_files)

# Configurar períodos
data_obj.period_list = np.logspace(
    np.log10(period_min), 
    np.log10(period_max), 
    n_periods
)

# Configurar errores
data_obj.error_type_z = 'egbert'  # Método de Egbert para errores
data_obj.error_type_tipper = 'abs'
data_obj.error_value_z = 0.05  # 5% error mínimo
data_obj.inv_mode = '1'  # Modo 2D (solo off-diagonal)

# Escribir archivo de datos
data_file = data_obj.write_data_file(
    save_path=output_path,
    fn_basename=basename
)
print(f"   ✓ Archivo de datos: {os.path.basename(data_file)}")

# 3. Crear modelo inicial
print("\n3. Creando modelo inicial...")
model_obj = modem.Model(
    Data=data_obj,
    save_path=output_path
)

model_obj.res_initial_value = res_initial
model_obj.cell_size_east = cell_size_x
model_obj.n_layers = n_layers
model_obj.z1_layer = z1_layer

# Crear malla
model_obj.make_mesh()

# Escribir modelo
model_file = model_obj.write_model_file(
    save_path=output_path,
    fn_basename=basename
)
print(f"   ✓ Archivo de modelo: {os.path.basename(model_file)}")
print(f"   Dimensiones: {model_obj.nodes_east.size} x {model_obj.nodes_z.size}")

# 4. Crear archivo de covarianza (suavizado)
print("\n4. Creando archivo de covarianza...")
cov_obj = modem.Covariance()
cov_obj.smoothing_east = smoothing_x
cov_obj.smoothing_north = smoothing_x
cov_obj.smoothing_z = smoothing_z

cov_file = cov_obj.write_covariance_file(
    save_path=output_path,
    fn_basename=basename
)
print(f"   ✓ Archivo de covarianza: {os.path.basename(cov_file)}")

# 5. Resumen
print("\n" + "="*60)
print("✓ ARCHIVOS GENERADOS EXITOSAMENTE")
print("="*60)
print(f"\nDirectorio: {output_path}")
print(f"  - {basename}.dat    (datos observados)")
print(f"  - {basename}.rho    (modelo inicial)")
print(f"  - {basename}.cov    (covarianza)")
print(f"\nEstaciones: {len(edi_files)}")
print(f"Frecuencias: {n_periods}")
print(f"Resistividad inicial: {res_initial} Ωm")
print(f"\n📤 Siguiente paso: Subir estos archivos a MAGMA")
print(f"   scp {output_path}/* magma:~/inversion_2d/")