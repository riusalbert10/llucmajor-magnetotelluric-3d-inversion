#!/usr/bin/env python3
# Usar Python iterprete .\mtpy_env_windows\Scripts\python.exe
"""
Script para explorar archivos EDI de datos magnetotelúricos
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mtpy.core.mt import MT

# Directorio con archivos EDI (compatible con Windows y WSL)
if sys.platform == 'win32':
    edi_dir = r'C:\Users\alber\TFG\LLUCMAJOR_DADES_edi'
else:
    edi_dir = '/mnt/c/Users/alber/TFG/LLUCMAJOR_DADES_edi'

# Obtener lista de archivos EDI
edi_files = sorted([f for f in os.listdir(edi_dir) if f.endswith('.edi')])
print(f"Total de archivos EDI encontrados: {len(edi_files)}\n")

# Leer todos los archivos y extraer información básica
stations_info = []

for edi_file in edi_files:
    edi_path = os.path.join(edi_dir, edi_file)

    try:
        # Leer archivo EDI con MTpy
        mt_obj = MT()
        mt_obj.read(edi_path)

        # Extraer información
        station = mt_obj.station
        lat = mt_obj.latitude
        lon = mt_obj.longitude
        elev = mt_obj.elevation if hasattr(mt_obj, 'elevation') else 0
        n_freqs = len(mt_obj.frequency)
        freq_range = f"{mt_obj.frequency.min():.3f} - {mt_obj.frequency.max():.1f} Hz"

        stations_info.append({
            'file': edi_file,
            'station': station,
            'lat': lat,
            'lon': lon,
            'elev': elev,
            'n_freqs': n_freqs,
            'freq_range': freq_range
        })

        print(f"{edi_file:15s} | Station: {station:10s} | Lat: {lat:10.6f} | Lon: {lon:10.6f} | Freqs: {n_freqs:2d}")

    except Exception as e:
        print(f"Error leyendo {edi_file}: {str(e)}")

print(f"\n{'='*80}")
print("RESUMEN DE LA CAMPAÑA")
print(f"{'='*80}")
print(f"Total estaciones: {len(stations_info)}")

# Calcular estadísticas
if stations_info:
    lats = [s['lat'] for s in stations_info]
    lons = [s['lon'] for s in stations_info]

    print(f"\nRango de coordenadas:")
    print(f"  Latitud:  {min(lats):.6f} - {max(lats):.6f}")
    print(f"  Longitud: {min(lons):.6f} - {max(lons):.6f}")

    # Calcular área aproximada
    lat_range = (max(lats) - min(lats)) * 111  # km
    lon_range = (max(lons) - min(lons)) * 111 * np.cos(np.radians(np.mean(lats)))  # km
    print(f"\nDimensiones del área:")
    print(f"  N-S: ~{lat_range:.2f} km")
    print(f"  E-W: ~{lon_range:.2f} km")

    print(f"\nFrecuencias:")
    print(f"  {stations_info[0]['freq_range']}")
    print(f"  Número de frecuencias: {stations_info[0]['n_freqs']}")

print("\n" + "="*80)
print("Exploración completada!")
