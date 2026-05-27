"""
paralelo.py
Processa todas as imagens do dataset em paralelo usando multiprocessing.Pool.

Testa automaticamente 4 configurações de processos: 2, 4, 8 e 12.
Para cada configuração:
  1. Divide as imagens em lotes iguais entre os processos
  2. Cada processo carrega o modelo YOLOv8 e processa seu lote de forma independente
  3. O processo principal reúne todos os resultados
  4. Salva tempo e speedup em tempos_paralelos.csv

Uso:
    python paralelo.py

⚠️  Windows: o bloco `if __name__ == "__main__":` é obrigatório.
    O Windows usa o método "spawn" para criar processos (diferente do Linux
    que usa "fork"), e sem esse bloco o script seria executado infinitamente
    em cada processo filho.
"""

import os
import csv
import time
import sys
import multiprocessing as mp
from pathlib import Path

import cv2
from ultralytics import YOLO

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURAÇÕES — ajuste aqui conforme necessário
# ──────────────────────────────────────────────────────────────────────────────
DATASET_DIR     = "DATASET-COVID-MASK"
MODELO_PATH     = r"C:\Users\Álex AlmSan\runs\detect\runs\detect\mask_detector\weights\best.pt"
CONFIANCA       = 0.4
SAIDA_CSV       = "resultados_paralelo.csv"
SAIDA_TEMPOS    = "tempos_paralelos.csv"
SAIDA_TEMPO_REF = "tempo_serial.txt"            # gerado pelo serial.py

CONFIGURACOES_PROCESSOS = [2, 4, 8, 12]        # configurações testadas

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


def processar_lote(args: tuple) -> list[dict]:
    """
    Função executada em cada processo filho.
    Recebe (id_lote, lista_de_caminhos) e retorna lista de resultados.

    Cada processo filho carrega o modelo YOLO de forma independente —
    não há compartilhamento de memória do modelo entre processos.
    """
    id_lote, caminhos = args

    # Carrega o modelo dentro do processo filho
    modelo = YOLO(MODELO_PATH)
    resultados = []

    for caminho in caminhos:
        r = modelo(caminho, conf=CONFIANCA, verbose=False)
        contagem = {"with_mask": 0, "without_mask": 0, "mask_weared_incorrect": 0}

        for resultado in r:
            for box in resultado.boxes:
                classe_id = int(box.cls[0])
                nome = CLASSES.get(classe_id)
                if nome:
                    contagem[nome] += 1

        contagem["total"] = sum(contagem.values())
        contagem["imagem"] = Path(caminho).name
        resultados.append(contagem)

    print(f"  [Processo {id_lote}] ✓ {len(caminhos)} imagens concluídas")
    return resultados


def dividir_lotes(lista: list, n: int) -> list[list]:
    """Divide lista em n lotes o mais iguais possível."""
    tam = max(1, len(lista) // n)
    return [lista[i:i + tam] for i in range(0, len(lista), tam)]


def ler_tempo_serial() -> float | None:
    """Lê o tempo serial salvo pelo serial.py para calcular speedup."""
    try:
        return float(Path(SAIDA_TEMPO_REF).read_text().strip())
    except Exception:
        return None


def salvar_csv_resultados(resultados: list[dict]):
    campos = ["imagem", "with_mask", "without_mask", "mask_weared_incorrect", "total"]
    with open(SAIDA_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        writer.writerows(resultados)
    print(f"[CSV] Resultados salvos em: {SAIDA_CSV}")


def salvar_csv_tempos(registros: list[dict]):
    campos = ["processos", "imagens_por_processo", "tempo_s", "speedup", "eficiencia_pct"]
    with open(SAIDA_TEMPOS, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        writer.writerows(registros)
    print(f"[CSV] Tempos salvos em: {SAIDA_TEMPOS}")


def main():
    if not Path(MODELO_PATH).exists():
        print(f"[ERRO] Modelo não encontrado em '{MODELO_PATH}'")
        print("       Execute primeiro: python treinar.py")
        sys.exit(1)

    imagens = coletar_imagens(DATASET_DIR)
    if not imagens:
        print(f"[ERRO] Nenhuma imagem encontrada em: {DATASET_DIR}")
        sys.exit(1)

    t_serial = ler_tempo_serial()
    if t_serial:
        print(f"[REF]  Tempo serial lido: {t_serial:.2f}s")
    else:
        print("[AVISO] tempo_serial.txt não encontrado — speedup não será calculado.")
        print("        Execute serial.py antes para obter o speedup.\n")

    registros_tempo = []
    ultimos_resultados = []

    print(f"[PARALELO] {len(imagens)} imagens | testando: {CONFIGURACOES_PROCESSOS} processos\n")

    for n_proc in CONFIGURACOES_PROCESSOS:
        # Não faz sentido usar mais processos que imagens
        n_proc_real = min(n_proc, len(imagens))

        lotes = dividir_lotes(imagens, n_proc_real)
        args  = [(i, lote) for i, lote in enumerate(lotes)]

        imgs_por_proc = len(imagens) // n_proc_real

        print(f"─── {n_proc_real} processos (~{imgs_por_proc} imgs cada) ───")
        inicio = time.perf_counter()

        with mp.Pool(processes=n_proc_real) as pool:
            lotes_resultado = pool.map(processar_lote, args)

        tempo = time.perf_counter() - inicio

        # Achata lista de listas
        resultados = [item for sub in lotes_resultado for item in sub]
        ultimos_resultados = resultados   # guarda o último para salvar no CSV

        # Calcula speedup e eficiência
        if t_serial:
            speedup    = t_serial / tempo
            eficiencia = speedup / n_proc_real * 100
        else:
            speedup    = 0.0
            eficiencia = 0.0

        print(f"    Tempo     : {tempo:.2f}s")
        if t_serial:
            print(f"    Speedup   : {speedup:.2f}x")
            print(f"    Eficiência: {eficiencia:.1f}%")
        print()

        registros_tempo.append({
            "processos":           n_proc_real,
            "imagens_por_processo": imgs_por_proc,
            "tempo_s":             round(tempo, 4),
            "speedup":             round(speedup, 4),
            "eficiencia_pct":      round(eficiencia, 2),
        })

    salvar_csv_resultados(ultimos_resultados)
    salvar_csv_tempos(registros_tempo)

    # ── Resumo final ─────────────────────────────────────────────────────────
    print("=" * 50)
    print(f"{'Processos':>10} {'Tempo (s)':>12} {'Speedup':>10} {'Eficiência':>12}")
    print("-" * 50)
    for r in registros_tempo:
        print(f"{r['processos']:>10} {r['tempo_s']:>12.2f} {r['speedup']:>10.2f}x {r['eficiencia_pct']:>11.1f}%")
    print("=" * 50)


if __name__ == "__main__":
    main()
