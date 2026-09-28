
import csv
import glob
import os
import sys
from collections import Counter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POSE_DIR = os.path.join(BASE_DIR, "output", "pose")
ROTULOS_DIR = os.path.join(BASE_DIR, "data", "rotulos")
ARQUIVOS_ROTULO = ["urfall-cam0-falls.csv", "urfall-cam0-adls.csv"]
CLASSE = {-1: 0, 0: 1, 1: 2}
NOME_CLASSE = {0: "estavel", 1: "desequilibrio", 2: "no chao"}
NOVAS = ["label", "classe"]


def carregar_rotulos():
    rot, achados = {}, []
    for nome in ARQUIVOS_ROTULO:
        caminho = os.path.join(ROTULOS_DIR, nome)
        if not os.path.exists(caminho):
            continue
        achados.append(nome)
        with open(caminho, newline="") as f:
            for r in csv.reader(f):
                if len(r) < 3:
                    continue
                rot.setdefault(r[0], {})[int(r[1])] = int(r[2])
    return rot, achados


def seq_id_do_arquivo(caminho):
    nome = os.path.basename(caminho)[:-4]      
    for sufixo in ("-cam0-rgb", "-cam1-rgb"):
        if nome.endswith(sufixo):
            return nome[: -len(sufixo)]
    return nome


def processar(caminho, rot_seq):
    with open(caminho, newline="") as f:
        linhas = list(csv.reader(f))

    cabecalho, dados = linhas[0], linhas[1:]

    # remove as colunas antigas, se ja existirem, e reinsere apos "detectado"
    manter = [i for i, c in enumerate(cabecalho) if c not in NOVAS]
    base = [cabecalho[i] for i in manter]
    pos = base.index("detectado") + 1 if "detectado" in base else 1
    novo_cabecalho = base[:pos] + NOVAS + base[pos:]

    i_frame = base.index("frame")
    saida, sem_rotulo, dist = [], 0, Counter()

    for linha in dados:
        valores = [linha[i] for i in manter]
        frame = int(valores[i_frame])
        if frame in rot_seq:
            label = rot_seq[frame]
            classe = CLASSE[label]
            dist[classe] += 1
            novos = [str(label), str(classe)]
        else:
            sem_rotulo += 1
            novos = ["", ""]
        saida.append(valores[:pos] + novos + valores[pos:])

    tmp = caminho + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(novo_cabecalho)
        w.writerows(saida)
    os.replace(tmp, caminho)

    return len(dados), sem_rotulo, dist


def main():
    rotulos, achados = carregar_rotulos()
    if not rotulos:
        sys.exit("Nenhum arquivo de rotulo encontrado em " + ROTULOS_DIR +
                 "\nEsperado: " + " ou ".join(ARQUIVOS_ROTULO))

    arquivos = sorted(glob.glob(os.path.join(POSE_DIR, "*.csv")))
    if not arquivos:
        sys.exit("Nenhum CSV de pose em " + POSE_DIR + " -- rode o main.py antes.")

    print("rotulos:", ", ".join(achados), f"({len(rotulos)} sequencias)")
    print(f"pose   : {len(arquivos)} arquivos\n")

    total, sem_total, sem_rotulo_arq = Counter(), 0, []

    for caminho in arquivos:
        seq = seq_id_do_arquivo(caminho)
        nome = os.path.basename(caminho)

        if seq not in rotulos:
            sem_rotulo_arq.append(seq)
            print(f"  {nome:28s} SEM ROTULO no URFD -- pulado")
            continue

        n, sem, dist = processar(caminho, rotulos[seq])
        total.update(dist)
        sem_total += sem

        resumo = " ".join(f"{NOME_CLASSE[c]}={dist[c]}" for c in (0, 1, 2))
        alerta = f"  <-- {sem} frame(s) sem rotulo" if sem else ""
        print(f"  {nome:28s} {n:4d} frames | {resumo}{alerta}")

    print()
    soma = sum(total.values())
    print("TOTAL rotulado:", soma, "frames")
    for c in (0, 1, 2):
        pct = 100 * total[c] / soma if soma else 0
        print(f"  classe {c} ({NOME_CLASSE[c]:13s}): {total[c]:5d}  ({pct:4.1f}%)")
    if sem_total:
        print(f"\nframes sem rotulo correspondente: {sem_total} (label/classe vazios)")
    if sem_rotulo_arq:
        print("sequencias sem entrada no URFD:", ", ".join(sem_rotulo_arq))


if __name__ == "__main__":
    main()
