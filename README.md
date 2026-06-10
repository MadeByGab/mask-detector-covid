# 😷 Projeto Programação CD — Detecção de Máscara COVID-19

## 📋 Sobre o Dataset

Este projeto utiliza o **Face Mask Detection Dataset**, um conjunto de dados para classificação do uso correto de máscara facial, composto por imagens de pessoas em ambientes públicos capturadas durante o período da pandemia de COVID-19.

O dataset contém imagens anotadas no formato **YOLO**, com três categorias de uso de máscara, permitindo identificar não apenas a ausência da máscara, mas também o uso incorreto (abaixo do nariz, no queixo etc.).

## 📊 Informações do Dataset

| Informação              | Detalhe                                                                                                                                                     |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 🗂️ Subconjuntos         | Full\_Dataset, Training, Validation                                                                                                                         |
| 😷 Classes               | 3 (mask, no\_mask, mask\_not\_in\_position)                                                                                                                 |
| 📐 Formato das anotações | YOLO (arquivos `.txt` com bounding boxes normalizados)                                                                                                      |
| 📁 Formato das imagens   | JPEG / PNG                                                                                                                                                  |
| 🔗 Fonte                 | Tamanho total de 3.51GB / 26.2k de arquivos. [Covid Face-Mask Monitoring Dataset](https://www.kaggle.com/datasets/jishan900/covid-facemask-monitoring-dataset) |

### Classes detectadas

| ID | Classe                 | Descrição                                          |
| --- | ---------------------- | -------------------------------------------------- |
| 0  | `mask`                 | Máscara usada corretamente (cobrindo nariz e boca) |
| 1  | `no_mask`              | Pessoa sem máscara                                 |
| 2  | `mask_not_in_position` | Máscara usada de forma incorreta                   |

## 💻 Hardware Utilizado nos Testes

| Componente | Detalhe                        |
| ---------- | ------------------------------ |
| 🖥️ CPU     | AMD Ryzen 5 5600GT             |
| 🧵 Núcleos | 6 núcleos físicos / 12 threads |
| 🎮 GPU     | AMD Radeon (integrada)         |
| ⚙️ Execução | CPU only (sem suporte a CUDA)  |

> ⚠️ O YOLO utiliza CUDA para aceleração em GPU, que é exclusivo de placas **NVIDIA**. Por esse motivo, todos os testes foram executados inteiramente na CPU, o que impacta diretamente o tempo de processamento e o comportamento do paralelismo.

## 📁 Arquivos do Projeto

| Arquivo                        | Descrição                                                     |
| ------------------------------ | ------------------------------------------------------------- |
| `treinar.py`                   | Treina o modelo YOLOv8 com o dataset (executar uma vez)       |
| `serial.py`                    | Processa todas as imagens sequencialmente com batch inference  |
| `paralelo.py`                  | Processa as imagens em paralelo usando 2, 4, 8 e 12 processos |
| `DATASET-COVID-MASK/data.yaml` | Configuração do dataset para o YOLO                           |

## 🔍 Como os Scripts Funcionam

### Captura das Imagens

Ambos os scripts percorrem automaticamente todas as subpastas do dataset usando `os.walk()`, coletando o caminho de cada imagem `.jpg` ou `.png` encontrada:

```python
def coletar_imagens(base_dir):
    for root, _, files in os.walk(base_dir):
        for f in sorted(files):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                yield os.path.join(root, f)
```

### `treinar.py` — Treinamento do Modelo

Usa o modelo base `yolov8n.pt` (YOLOv8 nano) pré-treinado no dataset COCO e realiza **transfer learning** com o dataset de máscaras.

Parâmetros principais:

- **epochs**: número de ciclos completos de treinamento (padrão: 10)
- **batch**: imagens processadas por iteração (padrão: 8)
- **patience**: early stopping — para automaticamente se o modelo não melhorar por N épocas seguidas
- **device**: `"cpu"` — GPU AMD não é compatível com CUDA/YOLO

O modelo treinado é salvo em `runs/detect/mask_detector/weights/best.pt`.

---

### `serial.py` — Processamento Sequencial

Processa as imagens em lotes sequenciais usando **batch inference** do YOLO. Para cada lote:

1. Carrega o modelo YOLOv8 treinado uma única vez antes do loop
2. Envia `BATCH_SIZE` imagens por chamada ao modelo
3. Lê os bounding boxes retornados e identifica a classe de cada detecção
4. Contabiliza quantas detecções de cada classe existem
5. Armazena os resultados em memória e salva ao final

Ao final, salva o tempo total em `tempo_serial.txt` e todos os resultados em `resultados_serial.csv`.

---

### `paralelo.py` — Processamento Paralelo

Distribui as imagens entre múltiplos processos usando `multiprocessing.Pool` com `initializer` — o modelo YOLO é carregado **uma vez por processo**, eliminando o custo de recarga a cada tarefa. Usa `imap_unordered` para processar resultados à medida que chegam.

| Processos | Imagens por processo (aprox.) |
| --------- | ----------------------------- |
| 2         | 6558                          |
| 4         | 3279                          |
| 8         | 1640                          |
| 12        | 1093                          |

## 📈 Resultados Obtidos

Testes executados com **13100 imagens** na máquina descrita acima (CPU only):

| Processos | Tempo (s) | Speedup | Eficiência |
| --------- | --------- | ------- | ---------- |
| Serial    | 315.15s   | 1.00x   | 100%       |
| 2         | 230.06s   | 1.37x   | 68.5%      |
| 4         | 189.36s   | 1.66x   | 41.6%      |
| 8         | 212.05s   | 1.49x   | 18.6%      |
| 12        | 224.14s   | 1.41x   | 11.7%      |

## 🆚 Comparativo: Código Original vs Código Otimizado

Esta seção compara o desempenho do código original (executado em outro PC com hardware equivalente) com o código otimizado desenvolvido neste projeto.
---

### Serial: Original vs Otimizado

| Versão    | Tempo (s) | Throughput     |
| --------- | --------- | -------------- |
| Original  | 506.39s   | 25.9 imgs/s    |
| Otimizado | 315.15s   | 41.6 imgs/s    |
| **Ganho** | **−191s** | **+60% mais rápido** |

---

### Paralelo: Original vs Otimizado

| Processos | Original (s) | Speedup Original | Otimizado (s) | Speedup Otimizado |
| --------- | ------------ | ---------------- | ------------- | ----------------- |
| 2         | 292.44s      | 1.73x            | 230.06s       | 1.37x             |
| 4         | 275.53s      | 1.84x            | 189.36s       | 1.66x             |
| 8         | 291.94s      | 1.73x            | 212.05s       | 1.49x             |
| 12        | 340.17s      | 1.49x            | 224.14s       | 1.41x             |

---

### Análise do comparativo

O código otimizado reduziu o tempo serial em **~38%** (de 506s para 315s), graças principalmente ao batch inference que reduz drasticamente o overhead de chamadas ao modelo YOLO.

No paralelo, o tempo absoluto também melhorou em todas as configurações — com 4 processos, por exemplo, caiu de 275s para 189s. O speedup relativo ao serial ficou ligeiramente menor no código otimizado porque o serial também foi muito acelerado, elevando a baseline de comparação.

O ponto ótimo em ambas as versões foi **4 processos**, confirmando que este é o número ideal para o hardware utilizado (6 núcleos físicos). Acima disso, a contenção de núcleos e o overhead de IPC superam os ganhos do paralelismo.

## 🧠 Análise do Comportamento da CPU

**Melhor resultado: 4 processos com speedup de 1.66x**

O ganho máximo foi atingido com 4 processos, reduzindo o tempo de 315s para 189s. Isso ocorre porque o Ryzen 5 5600GT possui 6 núcleos físicos — com 4 processos, cada um tem núcleos suficientes disponíveis sem concorrência excessiva.

**Por que piora a partir de 8 processos?**

Com 8 e 12 processos, o speedup cai para 1.49x e 1.41x respectivamente. O modelo YOLOv8 já utiliza internamente múltiplas threads da CPU. Com muitos processos paralelos, eles competem pelos mesmos núcleos físicos, gerando **contenção de recursos** e reduzindo o ganho esperado. O overhead de criação e comunicação entre processos também passa a pesar mais.

**Conclusão**

Em ambientes sem GPU NVIDIA, o paralelismo via `multiprocessing` para inferência YOLO apresenta ganhos modestos e não lineares. O ponto ótimo neste hardware foi **4 processos (1.66x de speedup)**. Para ganhos expressivos, seria necessário hardware com GPU NVIDIA e uso de CUDA.

## ⚙️ Como Executar

### 1. Instalar dependências

```
pip install ultralytics opencv-python pandas numpy
```

### 2. Configurar os caminhos

Abra `serial.py` e `paralelo.py` e ajuste as variáveis no topo:

```python
DATASET_DIR = "DATASET-COVID-MASK"
MODELO_PATH = "runs/detect/mask_detector/weights/best.pt"
```

### 3. Treinar o modelo

```
python treinar.py
```

> ⏱ O treinamento leva entre 1 e 2 horas em CPU (10 épocas).

### 4. Executar

```
python serial.py
python paralelo.py
```

## 📤 Arquivos Gerados

| Arquivo                   | Conteúdo                                                        |
| ------------------------- | --------------------------------------------------------------- |
| `tempo_serial.txt`        | Tempo de execução do processamento serial (em segundos)         |
| `resultados_serial.csv`   | Contagem de detecções por imagem (serial)                       |
| `resultados_paralelo.csv` | Contagem de detecções por imagem (paralelo, última config.)     |
| `tempos_paralelos.csv`    | Tempo, speedup e eficiência para cada configuração de processos |

## 📐 Métricas de Desempenho

| Métrica    | Fórmula                 | Significado                                 |
| ---------- | ----------------------- | ------------------------------------------- |
| Speedup    | T\_serial / T\_paralelo | Quantas vezes ficou mais rápido             |
| Eficiência | Speedup / N\_processos  | Aproveitamento de cada núcleo (ideal = 1.0) |
| Throughput | imagens / segundo       | Capacidade de processamento                 |

## ⬇️ Como baixar o dataset

As imagens não estão incluídas neste repositório devido ao tamanho dos arquivos.

Faça o download pelo link abaixo e extraia mantendo a estrutura de pastas original:

🔗 **[Covid Face-Mask Monitoring Dataset — Kaggle](https://www.kaggle.com/datasets/jishan900/covid-facemask-monitoring-dataset)**

A estrutura esperada após a extração:

```
DATASET-COVID-MASK/
├── Full_Dataset/
│   ├── images/
│   └── labels/
├── Training/
│   ├── images/
│   └── labels/
├── Validation/
│   ├── images/
│   └── labels/
├── classes.txt
└── data.yaml
```


GRÁFICOS ->

<img width="1177" height="732" alt="grafico_speedup" src="https://github.com/user-attachments/assets/7de618aa-a40c-470c-b770-b23033f410a4" />



<img width="1182" height="734" alt="grafico_tempo" src="https://github.com/user-attachments/assets/1f899ec0-6e11-417f-b3a7-ea24199982f0" />



<img width="1177" height="732" alt="grafico_eficiencia" src="https://github.com/user-attachments/assets/3fb127b4-f748-4a31-b230-a9df1e032476" />



