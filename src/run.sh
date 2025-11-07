#!/bin/bash
DATASET_PATH="../dataset/webspam_wc_normalized_unigram.svm"
MODEL_PATH="../models/webspam_tron.npy"
RESULTS_PATH="results_comparison.json"

C=1.0
PARTITIONS=32
MAX_OUTER_ITER=20
MAX_ITER_MLLIB=100
COALESCE=32

echo "=============================================="
echo "     EXPERIMENTO DISTRIBUIDO SPARK - TRON"
echo "=============================================="
echo "Dataset: $DATASET_PATH"
echo "Modelo:  $MODEL_PATH"
echo "----------------------------------------------"

echo "[1] Verificando carga del dataset..."
python data_loader.py --data "$DATASET_PATH"
if [ $? -ne 0 ]; then
  echo "Error cargando el dataset. Abortando."
  exit 1
fi
echo "OK: Dataset cargado correctamente."
echo "----------------------------------------------"

if [ ! -f "$MODEL_PATH" ]; then
  echo "[2] Entrenando modelo TRON..."
  python distributed_tron.py --data "$DATASET_PATH" \
      --C $C --partitions $PARTITIONS --max_outer_iter $MAX_OUTER_ITER \
      --coalesce $COALESCE --eval --save "$MODEL_PATH"
else
  echo "[2] Modelo TRON encontrado. Saltando entrenamiento."
fi
echo "----------------------------------------------"

echo "[3] Ejecutando comparación TRON vs MLlib..."
python evaluation.py \
    --data "$DATASET_PATH" \
    --C $C \
    --partitions $PARTITIONS \
    --max_outer_iter $MAX_OUTER_ITER \
    --max_iter_mllib $MAX_ITER_MLLIB \
    --coalesce $COALESCE \
    --reuse_model
if [ $? -ne 0 ]; then
  echo "Error ejecutando comparación. Abortando."
  exit 1
fi
echo "----------------------------------------------"

echo "[4] Resultados finales:"
cat "$RESULTS_PATH"
echo
echo "=============================================="
echo "Experimento completado correctamente."
echo "=============================================="
