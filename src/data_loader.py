from pyspark import SparkContext
from pyspark.mllib.linalg import Vectors

def get_spark_context():
    """Crea o recupera un SparkContext y ajusta el nivel de logs."""
    sc = SparkContext.getOrCreate()
    sc.setLogLevel("WARN")  # Reduce la verbosidad de los mensajes
    return sc

def parse_svm_line(line, fixed_dim=None):
    """
    Parsea una línea en formato tipo SVMlight y la convierte en (label, vector).
    - label: primer valor numérico de la línea.
    - vector: representa las características como vector disperso.
    """
    parts = line.strip().split()
    # Verifica que la línea comience con una etiqueta válida
    if not parts or not parts[0].replace('.', '', 1).replace('-', '', 1).isdigit():
        return None

    label = float(parts[0])  # Etiqueta del ejemplo
    indices, values = [], []

    # Procesa los pares index:value
    for item in parts[1:]:
        try:
            index, value = item.split(":")
            indices.append(int(index) - 1)  # Ajuste a índice base 0
            values.append(float(value))
        except ValueError:
            continue  # Ignora elementos mal formados

    # Determina la dimensión del vector (fijada o calculada)
    dim = fixed_dim if fixed_dim is not None else (max(indices) + 1 if indices else 0)

    # Construye vector disperso
    vector = Vectors.sparse(dim, indices, values)
    return (label, vector)

def load_svm_data(file_path, partitions=32):
    """
    Carga datos en formato SVMlight desde un archivo.
    Devuelve un RDD de elementos (label, vector).
    """
    sc = get_spark_context()
    raw_rdd = sc.textFile(file_path, minPartitions=partitions)

    # Calcula la dimensión global máxima revisando todas las líneas del archivo
    dim_max = raw_rdd.map(
        lambda l: max([int(i.split(":")[0]) for i in l.split()[1:] if ":" in i] or [0])
    ).max()  # type: ignore

    print(f"[INFO] Dimensión global detectada: {dim_max}")

    # Parsea cada línea, fijando la dimensión global detectada
    data_rdd = raw_rdd.map(
        lambda line: parse_svm_line(line, fixed_dim=dim_max)
    ).filter(lambda x: x is not None)

    data_rdd.cache()  # Optimiza accesos posteriores
    return data_rdd
