p = r"C:\Users\cuco2\OneDrive\Escritorio\Mis cosas\Analisis consumo electrico brasil CCEE\data\ddbb_scd\open\open_intervals.parquet"
import os

def parquet_head_tail(path: str):
    size = os.path.getsize(path)   # puede lanzar OSError si no existe
    head = tail = None
    if size >= 4:
        with open(path, "rb") as f:
            head = f.read(4)
            f.seek(size - 4, os.SEEK_SET)  # evita offset negativo
            tail = f.read(4)
    return size, head, tail

size, head, tail = parquet_head_tail(p)
print("size:", size, "head:", head, "tail:", tail)