"""
treinar.py
Treina o YOLOv8 com o dataset de máscaras COVID-19.

O modelo treinado é salvo em:
    runs/detect/mask_detector/weights/best.pt

Uso:
    python treinar.py
    python treinar.py --epochs 50 --modelo yolov8s.pt
"""

import argparse
from pathlib import Path
from ultralytics import YOLO

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURAÇÕES
# ──────────────────────────────────────────────────────────────────────────────
DATA_YAML   = "DATASET-COVID-MASK/data.yaml"  # usa o data.yaml do próprio dataset
MODELO_BASE = "yolov8n.pt"   # nano = mais rápido; yolov8s.pt = mais preciso
EPOCHS      = 50
IMG_SIZE    = 416
BATCH       = 8             # reduza para 4 se der erro de memória (Out of Memory)
PROJECT     = r"S:\Projeto_concorrente_distribuido\runs\detect"
NOME        = "mask_detector"
# ──────────────────────────────────────────────────────────────────────────────


def treinar(data: str, modelo_base: str, epochs: int, batch: int):
    print("=" * 50)
    print(f"  Dataset     : {data}")
    print(f"  Modelo base : {modelo_base}")
    print(f"  Épocas      : {epochs}")
    print(f"  Batch size  : {batch}")
    print("=" * 50)

    model = YOLO(modelo_base)

    model.train(
        data=data,
        epochs=epochs,
        imgsz=IMG_SIZE,
        batch=batch,
        project=PROJECT,
        name=NOME,
        exist_ok=True,    # sobrescreve execuções anteriores com o mesmo nome
        patience=10,      # early stopping: para se não melhorar por 10 épocas seguidas
        save=True,
        device="cpu",     # troque por device=0 se tiver GPU NVIDIA disponível
        workers=2,
    )

    best = Path(PROJECT) / NOME / "weights" / "best.pt"
    print(f"\n✅ Treinamento concluído!")
    print(f"   Modelo salvo em: {best}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treina YOLOv8 para detecção de máscara COVID-19")
    parser.add_argument("--data",   default=DATA_YAML)
    parser.add_argument("--modelo", default=MODELO_BASE, help="yolov8n.pt | yolov8s.pt | yolov8m.pt")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch",  type=int, default=BATCH)
    args = parser.parse_args()

    treinar(args.data, args.modelo, args.epochs, args.batch)
