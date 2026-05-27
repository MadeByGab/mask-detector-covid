"""
serial.py
Processa todas as imagens do dataset sequencialmente, uma por uma.

Para cada imagem:
  1. Carrega com cv2.imread()
  2. Roda inferência YOLOv8 (modelo treinado)
  3. Contabiliza detecções por classe (mask / no_mask / mask_not_in_position)
  4. Salva resultados em resultados_serial.csv
  5. Registra tempo total em tempo_serial.txt

Uso:
    python serial.py
"""

import os
import csv
import time
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURAÇÕES — ajuste aqui conforme necessário
# ──────────────────────────────────────────────────────────────────────────────
DATASET_DIR   = "DATASET-COVID-MASK"          # pasta raiz do dataset
MODELO_PATH   = r"C:\Users\Álex AlmSan\runs\detect\runs\detect\mask_detector\weights\best.pt"  # modelo treinado
CONFIANCA     = 0.4                           # limiar mínimo de confiança
SAIDA_CSV     = "resultados_serial.csv"
SAIDA_TEMPO   = "tempo_serial.txt"

CLASSES = {
    0: "with_mask",
    1: "without_mask",
    2: "mask_weared_incorrect",
}
# ──────────────────────────────────────────────────────────────────────────────


def coletar_imagens(base_dir: str) -> list[str]:
    """Percorre todas as subpastas e retorna caminhos de imagens."""
    imagens = []
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                imagens.append(os.path.join(root, f))
    return sorted(imagens)


def processar_imagem(modelo: YOLO, caminho: str) -> dict:
    """
    Roda inferência em uma imagem e retorna contagem por classe.
    Retorna dict com: imagem, with_mask, without_mask, mask_weared_incorrect, total
    """
    resultados = modelo(caminho, conf=CONFIANCA, verbose=False)

    contagem = {"with_mask": 0, "without_mask": 0, "mask_weared_incorrect": 0}

    for r in resultados:
        for box in r.boxes:
            classe_id = int(box.cls[0])
            nome_classe = CLASSES.get(classe_id)
            if nome_classe:
                contagem[nome_classe] += 1

    contagem["total"] = sum(contagem.values())
    contagem["imagem"] = Path(caminho).name
    return contagem


def salvar_csv(resultados: list[dict]):
    campos = ["imagem", "with_mask", "without_mask", "mask_weared_incorrect", "total"]
    with open(SAIDA_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        writer.writerows(resultados)
    print(f"[CSV] Resultados salvos em: {SAIDA_CSV}")


def main():
    # Verifica se o modelo treinado existe
    if not Path(MODELO_PATH).exists():
        print(f"[ERRO] Modelo não encontrado em '{MODELO_PATH}'")
        print("       Execute primeiro: python treinar.py")
        sys.exit(1)

    print(f"[MODELO] Carregando: {MODELO_PATH}")
    modelo = YOLO(MODELO_PATH)

    imagens = coletar_imagens(DATASET_DIR)
    if not imagens:
        print(f"[ERRO] Nenhuma imagem encontrada em: {DATASET_DIR}")
        sys.exit(1)

    print(f"[SERIAL] Processando {len(imagens)} imagens sequencialmente...")
    print("-" * 50)

    resultados = []
    inicio = time.perf_counter()

    for i, caminho in enumerate(imagens, 1):
        resultado = processar_imagem(modelo, caminho)
        resultados.append(resultado)

        # Progresso a cada 50 imagens
        if i % 50 == 0 or i == len(imagens):
            print(f"  Progresso: {i}/{len(imagens)} imagens", end="\r")

    tempo_total = time.perf_counter() - inicio

    print(f"\n[SERIAL] Concluído em {tempo_total:.2f}s")

    # Salva tempo
    with open(SAIDA_TEMPO, "w") as f:
        f.write(f"{tempo_total:.4f}\n")
    print(f"[TEMPO]  Salvo em: {SAIDA_TEMPO}")

    # Salva CSV
    salvar_csv(resultados)

    # ── Resumo ──────────────────────────────────────────────────────────────
    total_mask     = sum(r["with_mask"]                 for r in resultados)
    total_no_mask  = sum(r["without_mask"]              for r in resultados)
    total_incorreta = sum(r["mask_weared_incorrect"] for r in resultados)
    total_rostos   = total_mask + total_no_mask + total_incorreta

    print("\n" + "=" * 50)
    print(f"  Imagens processadas       : {len(imagens)}")
    print(f"  Total de rostos detectados: {total_rostos}")
    if total_rostos > 0:
        print(f"  Máscara correta           : {total_mask}  ({total_mask/total_rostos*100:.1f}%)")
        print(f"  Sem máscara               : {total_no_mask}  ({total_no_mask/total_rostos*100:.1f}%)")
        print(f"  Máscara incorreta         : {total_incorreta}  ({total_incorreta/total_rostos*100:.1f}%)")
    print(f"  Tempo total               : {tempo_total:.2f}s")
    print(f"  Throughput                : {len(imagens)/tempo_total:.1f} imgs/s")
    print("=" * 50)


if __name__ == "__main__":
    main()
