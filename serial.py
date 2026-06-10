import os
import time
import csv
from ultralytics import YOLO

DATASET_DIR  = "C:\\Users\\gabri\\Downloads\\mask-detector-covid-main"
MODELO_PATH  = "runs/detect/runs/detect/mask_detector/weights/best.pt"
CONFIANCA    = 0.4
BATCH_SIZE   = 16

CLASSES = {0: "mask", 1: "no_mask", 2: "mask_not_in_position"}

SAIDA_TEMPO      = "tempo_serial.txt"
SAIDA_RESULTADOS = "resultados_serial.csv"


def coletar_imagens(base_dir):
    for root, _, files in os.walk(base_dir):
        for f in sorted(files):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                yield os.path.join(root, f)


def processar_em_lotes(modelo, imagens, batch_size):
    resultados = []
    for inicio in range(0, len(imagens), batch_size):
        lote = imagens[inicio : inicio + batch_size]
        preds = modelo(lote, conf=CONFIANCA, verbose=False)
        for caminho, pred in zip(lote, preds):
            contagens = {nome: 0 for nome in CLASSES.values()}
            if pred.boxes is not None:
                for cls_id in pred.boxes.cls.int().tolist():
                    nome_classe = CLASSES.get(cls_id)
                    if nome_classe:
                        contagens[nome_classe] += 1
            resultados.append({
                "imagem":               os.path.basename(caminho),
                "mask":                 contagens["mask"],
                "no_mask":              contagens["no_mask"],
                "mask_not_in_position": contagens["mask_not_in_position"],
                "total":                sum(contagens.values()),
            })
    return resultados


def main():
    print(f"Carregando modelo: {MODELO_PATH}")
    modelo = YOLO(MODELO_PATH)

    print(f"Coletando imagens em: {DATASET_DIR}")
    imagens = sorted(coletar_imagens(DATASET_DIR))
    print(f"Total de imagens encontradas: {len(imagens)}")

    print(f"Iniciando processamento serial (batch_size={BATCH_SIZE})...")
    t_inicio = time.perf_counter()

    resultados = processar_em_lotes(modelo, imagens, BATCH_SIZE)

    t_fim = time.perf_counter()
    tempo_total = t_fim - t_inicio

    with open(SAIDA_TEMPO, "w") as f:
        f.write(f"{tempo_total:.4f}\n")
    print(f"Tempo total: {tempo_total:.2f}s  →  salvo em '{SAIDA_TEMPO}'")

    campos = ["imagem", "mask", "no_mask", "mask_not_in_position", "total"]
    with open(SAIDA_RESULTADOS, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        writer.writerows(resultados)
    print(f"Resultados salvos em '{SAIDA_RESULTADOS}'")


if __name__ == "__main__":
    main()
