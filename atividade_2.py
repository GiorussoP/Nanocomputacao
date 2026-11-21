#!/usr/bin/env python3
"""
Atividade 2 — Colágeno SHG: foundation models vs. algoritmos clássicos (BOCA).

Execução padrão (roda todas as etapas em ordem e pula o que já está pronto):
  python3 atividade_2.py

Opções úteis em CPU:
  python3 atividade_2.py --pular-cellpose
  python3 atividade_2.py --cellpose-limit 11
  python3 atividade_2.py --cellpose-max-dim 512
"""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.filters import gaussian, sato, threshold_otsu
from skimage.morphology import (
    binary_closing,
    binary_opening,
    disk,
    remove_small_objects,
    skeletonize,
)
from skimage.segmentation import find_boundaries
import tifffile


ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "raw" / "IMAGENS"
META_DIR = ROOT / "metadata"
ANN_SHG_DIR = ROOT / "annotations" / "BOCA" / "SHG"
PROC_DIR = ROOT / "processed" / "foundation_microscopy"
SEG_ROOT = PROC_DIR / "segmentations" / "BOCA"

IMAGES_CSV = META_DIR / "foundation_images.csv"
SPLIT_IMAGES_CSV = META_DIR / "split_boca.csv"

DIR_MICROSAM = SEG_ROOT / "SHG"
DIR_CELLPOSE_SAM = SEG_ROOT / "SHG_cellpose_sam"
DIR_CURVELET_FIRE = SEG_ROOT / "SHG_curvelet_fire"
DIR_SATO_OTSU = SEG_ROOT / "SHG_sato_otsu"

OUT_DIR = PROC_DIR / "atividade_2_shg"
OVERLAYS_DIR = OUT_DIR / "overlays"

METRICAS_IMG_CSV = OUT_DIR / "metricas_segmentacao_por_imagem.csv"
MORFO_FIBRAS_CSV = OUT_DIR / "morfometria_fibras_individuais.csv"
MORFO_IMG_CSV = OUT_DIR / "morfometria_resumo_por_imagem.csv"
TABELA_COMP_CSV = OUT_DIR / "tabela_comparativa_atividade_2.csv"
FIGURA_RESUMO_PNG = OUT_DIR / "resumo_comparativo_atividade_2.png"

METODOS_ORDEM = ["microsam", "cellpose_sam", "curvelet_fire", "sato_otsu"]
METODOS_LABELS = {
    "ground_truth": "Padrão-Ouro (Manual)",
    "microsam": "micro-sam",
    "cellpose_sam": "Cellpose-SAM",
    "curvelet_fire": "Curvelet + FIRE/CT-FIRE",
    "sato_otsu": "Sato + Otsu + Morfologia",
}


# ==============================================================================
# Utilitários compartilhados
# ==============================================================================


def load_2d(path: Path) -> np.ndarray:
    arr = np.squeeze(tifffile.imread(path))
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Esperada imagem 2D em {path}, mas shape={arr.shape}")
    return arr


def norm_percentile(
    img: np.ndarray, p_low: float = 1.0, p_high: float = 99.5
) -> np.ndarray:
    arr = img.astype(np.float32)
    lo, hi = np.percentile(arr, (p_low, p_high))
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def inferir_condicao(relpath: str) -> str:
    s = str(relpath)
    if "SEM CORAR" in s:
        return "SEM CORAR"
    if "CORADO" in s:
        return "CORADO"
    if "11-01-2025" in s:
        return "11-01-2025"
    return "UNKNOWN"


def carregar_dataset_shg() -> pd.DataFrame:
    """Carrega o inventário SHG de BOCA cruzado com o split sem vazamento e o gabarito."""
    if SPLIT_IMAGES_CSV.exists():
        df = pd.read_csv(SPLIT_IMAGES_CSV)
    elif IMAGES_CSV.exists():
        df = pd.read_csv(IMAGES_CSV)
        df["condicao"] = df["relative_path"].apply(inferir_condicao)
        df["split"] = "unassigned"
    else:
        raise FileNotFoundError(
            f"Nem {SPLIT_IMAGES_CSV} nem {IMAGES_CSV} foram encontrados."
        )

    shg = df[(df["group"] == "BOCA") & (df["modality"] == "SHG")].copy()
    shg["image_id"] = shg["image_id"].astype(str)
    if "condicao" not in shg.columns:
        shg["condicao"] = shg["relative_path"].apply(inferir_condicao)
    if "split" not in shg.columns:
        shg["split"] = "unassigned"

    raw_paths = []
    gt_paths = []
    has_gt = []
    for _, row in shg.iterrows():
        raw_p = RAW_DIR / str(row["relative_path"])
        if not raw_p.exists() and "path" in row and pd.notna(row["path"]):
            raw_p = Path(str(row["path"]))
        raw_paths.append(str(raw_p))

        gt_p = ANN_SHG_DIR / f"{row['image_id']}.tif"
        has_gt.append(gt_p.exists())
        gt_paths.append(str(gt_p) if gt_p.exists() else "")

    shg["raw_path"] = raw_paths
    shg["has_gt"] = has_gt
    shg["gt_path"] = gt_paths
    return shg.sort_values(["condicao", "image_id"]).reset_index(drop=True)


def caminho_mascara_metodo(metodo: str, image_id: str) -> Path:
    mapa = {
        "microsam": DIR_MICROSAM / f"{image_id}.tif",
        "cellpose_sam": DIR_CELLPOSE_SAM / f"{image_id}.tif",
        "curvelet_fire": DIR_CURVELET_FIRE / f"{image_id}.tif",
        "sato_otsu": DIR_SATO_OTSU / f"{image_id}.tif",
    }
    if metodo not in mapa:
        raise ValueError(f"Método desconhecido: {metodo}")
    return mapa[metodo]


def carregar_mascara_binaria(path: Path) -> np.ndarray:
    return (load_2d(path) > 0).astype(np.uint8)


# ==============================================================================
# Etapa 1: Segmentação (micro-sam, Cellpose-SAM e Baselines Clássicos)
# ==============================================================================


def _executar_cellpose_sam_batch(
    tarefas: list[tuple[str, Path, Path]],
    cellpose_python: str | None = None,
    max_dim_cpu: int = 512,
    use_gpu: bool = True,
) -> int:
    """Executa Cellpose-SAM com progresso em tempo real e redimensionamento adaptativo em CPU."""
    if not tarefas:
        return 0

    DIR_CELLPOSE_SAM.mkdir(parents=True, exist_ok=True)
    py_exec = cellpose_python or sys.executable

    worker_code = r"""
import json, os, sys, time
from pathlib import Path
import numpy as np
from skimage.transform import resize
import tifffile

tasks = json.loads(sys.argv[1])
use_gpu = sys.argv[2] == "1"
max_dim_cpu = int(sys.argv[3])

try:
    import torch
    from cellpose import models
except ImportError as exc:
    print(f"IMPORT_ERROR: {exc}", flush=True)
    sys.exit(2)

gpu_disp = bool(use_gpu and torch.cuda.is_available())
if not gpu_disp:
    n_cores = os.cpu_count() or 4
    torch.set_num_threads(max(1, min(8, n_cores)))

modo_txt = "GPU (CUDA) - resolução nativa" if gpu_disp else f"CPU - max_dim={max_dim_cpu or 'nativa'}"
print(f"  [Cellpose-SAM] Dispositivo: {modo_txt} | Carregando modelo...", flush=True)

model = None
for kwargs in (
    {"gpu": gpu_disp, "pretrained_model": "cpsam"},
    {"gpu": gpu_disp, "model_type": "cpsam"},
    {"gpu": gpu_disp},
):
    try:
        model = models.CellposeModel(**kwargs)
        break
    except TypeError:
        continue

if model is None:
    raise RuntimeError("Não foi possível instanciar CellposeModel.")

def norm_p(img):
    arr = np.squeeze(img).astype(np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    lo, hi = np.percentile(arr, (1.0, 99.5))
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)

total = len(tasks)
done = 0
for idx, item in enumerate(tasks, 1):
    t0 = time.perf_counter()
    img_orig = norm_p(tifffile.imread(item["raw"]))
    orig_h, orig_w = img_orig.shape

    redimensionado = False
    img_in = img_orig
    if not gpu_disp and max_dim_cpu > 0 and max(orig_h, orig_w) > max_dim_cpu:
        escala = max_dim_cpu / float(max(orig_h, orig_w))
        new_h = max(64, int(round(orig_h * escala)))
        new_w = max(64, int(round(orig_w * escala)))
        img_in = resize(
            img_orig, (new_h, new_w), order=1, preserve_range=True, anti_aliasing=True
        ).astype(np.float32)
        redimensionado = True

    try:
        res = model.eval(img_in, diameter=None, channels=[0, 0], tile_overlap=0.05)
    except TypeError:
        res = model.eval(img_in, diameter=None, channels=[0, 0])

    masks = res[0] if isinstance(res, tuple) else res
    if redimensionado:
        masks = resize(
            masks, (orig_h, orig_w), order=0, preserve_range=True, anti_aliasing=False
        )

    out_p = Path(item["out"])
    out_p.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(out_p, masks.astype(np.uint16))
    dt = time.perf_counter() - t0
    done += 1
    info_dim = f"{orig_h}x{orig_w} -> {img_in.shape[0]}x{img_in.shape[1]}" if redimensionado else f"{orig_h}x{orig_w}"
    print(f"  [Cellpose-SAM {idx}/{total}] {item['image_id']} ({info_dim}) concluído em {dt:.1f}s", flush=True)

print(f"DONE:{done}", flush=True)
"""
    payload = json.dumps(
        [{"image_id": iid, "raw": str(rp), "out": str(op)} for iid, rp, op in tarefas]
    )
    proc = subprocess.Popen(
        [
            py_exec,
            "-u",
            "-c",
            worker_code,
            payload,
            "1" if use_gpu else "0",
            str(max_dim_cpu),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    done = 0
    saida_linhas: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        line_clean = line.rstrip()
        saida_linhas.append(line_clean)
        if line_clean.startswith("IMPORT_ERROR:"):
            proc.wait()
            print(
                "[AVISO] Biblioteca 'cellpose' não encontrada no ambiente atual. "
                "Instale com 'pip install cellpose' ou informe --cellpose-python /caminho/python. "
                "As demais etapas continuarão normalmente."
            )
            return 0
        if line_clean.startswith("DONE:"):
            done = int(line_clean.split(":", 1)[1])
        else:
            print(line_clean, flush=True)

    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(
            "Erro ao executar Cellpose-SAM:\n" + "\n".join(saida_linhas[-25:])
        )
    return done


def _filtro_curvelet_direcional_2d(
    img_norm: np.ndarray, n_escalas: int = 3, n_angulos: int = 16
) -> np.ndarray:
    """
    Realce multiescala e direcional inspirado na Transformada Curvelet (cunhas polares no
    domínio da frequência) para estruturas fibrilares curvas com supressão de fundo isotrópico.
    """
    h, w = img_norm.shape
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    rho = np.hypot(fx, fy)
    theta = np.arctan2(fy, fx)

    f_img = np.fft.fft2(img_norm)

    limites_radiais = np.geomspace(0.025, 0.45, n_escalas + 1)
    angulos = np.linspace(0.0, np.pi, n_angulos, endpoint=False)
    largura_ang = (np.pi / n_angulos) * 1.35

    respostas_escala = []
    for s in range(n_escalas):
        r0, r1 = limites_radiais[s], limites_radiais[s + 1]
        r_mid = math.sqrt(r0 * r1)
        sigma_r = max((r1 - r0) * 0.55, 1e-4)
        w_radial = np.exp(-0.5 * ((rho - r_mid) / sigma_r) ** 2)
        w_radial[rho < 0.015] = 0.0

        respostas_ang = []
        for ang in angulos:
            diff = np.angle(np.exp(2j * (theta - ang))) * 0.5
            w_ang = np.exp(-0.5 * (diff / largura_ang) ** 2)
            subbanda = np.real(np.fft.ifft2(f_img * w_radial * w_ang))
            respostas_ang.append(np.maximum(subbanda, 0.0))

        stack_ang = np.stack(respostas_ang, axis=0)
        anisotropia = np.max(stack_ang, axis=0) - np.median(stack_ang, axis=0)
        anisotropia = np.maximum(anisotropia, 0.0)
        if anisotropia.max() > 0:
            anisotropia /= anisotropia.max()
        respostas_escala.append(anisotropia)

    combinado = np.mean(np.stack(respostas_escala, axis=0), axis=0)
    if combinado.max() > 0:
        combinado /= combinado.max()
    return combinado.astype(np.float32)


def segmentar_curvelet_fire(
    img_raw: np.ndarray,
    fator_limiar: float = 1.0,
    min_comprimento_px: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Baseline clássico (a): Curvelet Transform + extração/rastreamento de fibras estilo FIRE.
    Retorna (mascara_binaria, esqueleto_fibras).
    """
    img_n = norm_percentile(img_raw, 1.0, 99.5)
    img_suave = gaussian(img_n, sigma=0.7, preserve_range=True)

    resp_curvelet = _filtro_curvelet_direcional_2d(img_suave, n_escalas=3, n_angulos=16)

    vals_pos = resp_curvelet[resp_curvelet > 0.02]
    if vals_pos.size < 20:
        vazio = np.zeros_like(img_n, dtype=np.uint8)
        return vazio, vazio

    th_alto = float(threshold_otsu(vals_pos)) * fator_limiar
    th_baixo = 0.55 * th_alto

    sementes = resp_curvelet >= th_alto
    suporte = resp_curvelet >= th_baixo

    labels_sup, n_comp = ndi.label(suporte, structure=np.ones((3, 3), dtype=int))
    if n_comp == 0:
        vazio = np.zeros_like(img_n, dtype=np.uint8)
        return vazio, vazio

    ids_validos = np.unique(labels_sup[sementes])
    ids_validos = ids_validos[ids_validos > 0]
    mascara_bruta = np.isin(labels_sup, ids_validos)
    mascara_bruta = binary_closing(mascara_bruta, disk(1))
    mascara_bruta = remove_small_objects(mascara_bruta, min_size=20)

    skel = skeletonize(mascara_bruta)
    labels_skel, _ = ndi.label(skel, structure=np.ones((3, 3), dtype=int))
    tamanhos = np.bincount(labels_skel.ravel())
    manter_skel = np.zeros_like(tamanhos, dtype=bool)
    manter_skel[1:] = tamanhos[1:] >= min_comprimento_px
    skel_limpo = manter_skel[labels_skel]

    edt = ndi.distance_transform_edt(mascara_bruta)
    raio_medio = float(
        np.clip(np.median(edt[skel_limpo]) if np.any(skel_limpo) else 1.5, 1.0, 4.0)
    )
    mascara_final = ndi.binary_dilation(
        skel_limpo, structure=disk(int(round(raio_medio)))
    )
    mascara_final = (mascara_final & suporte) | ndi.binary_dilation(
        skel_limpo, structure=disk(1)
    )
    mascara_final = remove_small_objects(mascara_final, min_size=20)

    return mascara_final.astype(np.uint8), skeletonize(mascara_final).astype(np.uint8)


def segmentar_sato_otsu(
    img_raw: np.ndarray,
    fator_otsu: float = 1.0,
    min_tamanho_obj: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Baseline clássico (b): Sato multiescala + Otsu + morfologia + esqueletização.
    Retorna (mascara_binaria, esqueleto_fibras).
    """
    img_n = norm_percentile(img_raw, 1.0, 99.5)
    img_suave = gaussian(img_n, sigma=0.8, preserve_range=True)

    tubeness = sato(img_suave, sigmas=(1.0, 2.0, 3.0, 4.0), black_ridges=False)
    if tubeness.max() > 0:
        tubeness = tubeness / tubeness.max()

    vals_pos = tubeness[tubeness > 0.01]
    if vals_pos.size < 20:
        vazio = np.zeros_like(img_n, dtype=np.uint8)
        return vazio, vazio

    th = float(threshold_otsu(vals_pos)) * fator_otsu
    mask = tubeness >= th
    mask = binary_closing(mask, disk(1))
    mask = remove_small_objects(mask, min_size=min_tamanho_obj)
    mask = binary_opening(mask, disk(1))
    skel = skeletonize(mask)

    return mask.astype(np.uint8), skel.astype(np.uint8)


def executar_segmentacoes(
    df_shg: pd.DataFrame,
    cellpose_python: str | None = None,
    ctfire_dir: Path | None = None,
    fator_curvelet: float = 1.0,
    fator_sato: float = 1.0,
    pular_cellpose: bool = False,
    cellpose_limit: int | None = None,
    cellpose_max_dim: int = 512,
    force: bool = False,
) -> None:
    DIR_MICROSAM.mkdir(parents=True, exist_ok=True)
    DIR_CELLPOSE_SAM.mkdir(parents=True, exist_ok=True)
    DIR_CURVELET_FIRE.mkdir(parents=True, exist_ok=True)
    DIR_SATO_OTSU.mkdir(parents=True, exist_ok=True)

    # 1. Conferência micro-sam
    faltantes_msam = [
        r for _, r in df_shg.iterrows() if not (DIR_MICROSAM / f"{r['image_id']}.tif").exists()
    ]
    if faltantes_msam:
        print(
            f"[AVISO] {len(faltantes_msam)} máscara(s) micro-sam ausente(s) em {DIR_MICROSAM}."
        )
    else:
        print(f"[PULAR] micro-sam: {len(df_shg)}/{len(df_shg)} máscaras já disponíveis.")

    # 2. Cellpose-SAM
    if pular_cellpose:
        n_exist = sum(
            (DIR_CELLPOSE_SAM / f"{r['image_id']}.tif").exists()
            for _, r in df_shg.iterrows()
        )
        print(
            f"[PULAR] Cellpose-SAM ignorado via --pular-cellpose ({n_exist}/{len(df_shg)} já prontas)."
        )
    else:
        tarefas_cp = []
        df_ord = df_shg.sort_values(
            ["has_gt", "condicao", "image_id"], ascending=[False, True, True]
        )
        for _, row in df_ord.iterrows():
            out_p = DIR_CELLPOSE_SAM / f"{row['image_id']}.tif"
            if force or not out_p.exists():
                tarefas_cp.append((str(row["image_id"]), Path(str(row["raw_path"])), out_p))

        if cellpose_limit is not None and cellpose_limit > 0:
            tarefas_cp = tarefas_cp[:cellpose_limit]

        if tarefas_cp:
            print(f"[ETAPA 1a] Executando Cellpose-SAM em {len(tarefas_cp)} imagem(ns)...")
            geradas_cp = _executar_cellpose_sam_batch(
                tarefas_cp,
                cellpose_python=cellpose_python,
                max_dim_cpu=cellpose_max_dim,
            )
            print(f"  -> Cellpose-SAM: {geradas_cp} máscara(s) gerada(s).")
        else:
            print(f"[PULAR] Cellpose-SAM: {len(df_shg)}/{len(df_shg)} máscaras já disponíveis.")

    # 3. Baselines clássicos (Curvelet+FIRE e Sato+Otsu)
    pendentes_classicos = 0
    for _, row in df_shg.iterrows():
        iid = str(row["image_id"])
        raw_p = Path(str(row["raw_path"]))
        out_cf = DIR_CURVELET_FIRE / f"{iid}.tif"
        out_so = DIR_SATO_OTSU / f"{iid}.tif"

        need_cf = force or not out_cf.exists()
        need_so = force or not out_so.exists()
        if not need_cf and not need_so:
            continue

        pendentes_classicos += 1
        img_raw = load_2d(raw_p)

        if need_cf:
            ext_candidate = (ctfire_dir / f"{iid}.tif") if ctfire_dir else None
            if ext_candidate is not None and ext_candidate.exists():
                mask_cf = carregar_mascara_binaria(ext_candidate)
                skel_cf = skeletonize(mask_cf > 0).astype(np.uint8)
            else:
                mask_cf, skel_cf = segmentar_curvelet_fire(
                    img_raw, fator_limiar=fator_curvelet
                )
            tifffile.imwrite(out_cf, mask_cf)
            tifffile.imwrite(DIR_CURVELET_FIRE / f"{iid}_skel.tif", skel_cf)

        if need_so:
            mask_so, skel_so = segmentar_sato_otsu(img_raw, fator_otsu=fator_sato)
            tifffile.imwrite(out_so, mask_so)
            tifffile.imwrite(DIR_SATO_OTSU / f"{iid}_skel.tif", skel_so)

    if pendentes_classicos == 0:
        print(
            f"[PULAR] Baselines clássicos: {len(df_shg)}/{len(df_shg)} imagens já processadas."
        )
    else:
        print(
            f"[OK] Baselines clássicos concluídos para {pendentes_classicos} imagem(ns)."
        )


# ==============================================================================
# Etapa 2: Calibração opcional no treino e Avaliação (Dice, IoU, Cobertura)
# ==============================================================================


def calcular_metricas_binarias(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    p = pred > 0
    g = gt > 0
    tp = int(np.logical_and(p, g).sum())
    fp = int(np.logical_and(p, ~g).sum())
    fn = int(np.logical_and(~p, g).sum())

    denom_dice = 2 * tp + fp + fn
    denom_iou = tp + fp + fn

    dice = (2.0 * tp / denom_dice) if denom_dice > 0 else 1.0
    iou = (float(tp) / denom_iou) if denom_iou > 0 else 1.0
    precisao = (float(tp) / (tp + fp)) if (tp + fp) > 0 else (1.0 if fn == 0 else 0.0)
    revocacao = (float(tp) / (tp + fn)) if (tp + fn) > 0 else (1.0 if fp == 0 else 0.0)

    cob_pred = 100.0 * float(p.mean())
    cob_gt = 100.0 * float(g.mean())
    return {
        "dice": dice,
        "iou": iou,
        "precisao": precisao,
        "revocacao": revocacao,
        "cobertura_pred_pct": cob_pred,
        "cobertura_gt_pct": cob_gt,
        "erro_cobertura_abs_pp": abs(cob_pred - cob_gt),
    }


def calibrar_fatores_no_treino(df_shg: pd.DataFrame) -> tuple[float, float]:
    treino_gt = df_shg[(df_shg["split"] == "train") & (df_shg["has_gt"])].copy()
    if treino_gt.empty:
        return 1.0, 1.0

    grade = [0.75, 0.9, 1.0, 1.15, 1.3]
    melhor_cf, score_cf = 1.0, -1.0
    melhor_so, score_so = 1.0, -1.0

    cache = [
        (load_2d(Path(r["raw_path"])), carregar_mascara_binaria(Path(r["gt_path"])))
        for _, r in treino_gt.iterrows()
    ]

    for f in grade:
        dices_cf = [
            calcular_metricas_binarias(segmentar_curvelet_fire(img, fator_limiar=f)[0], gt)["dice"]
            for img, gt in cache
        ]
        m_cf = float(np.mean(dices_cf))
        if m_cf > score_cf:
            score_cf, melhor_cf = m_cf, f

        dices_so = [
            calcular_metricas_binarias(segmentar_sato_otsu(img, fator_otsu=f)[0], gt)["dice"]
            for img, gt in cache
        ]
        m_so = float(np.mean(dices_so))
        if m_so > score_so:
            score_so, melhor_so = m_so, f

    print(
        f"[CALIBRAÇÃO TREINO] fator_curvelet={melhor_cf:.2f} (Dice treino={score_cf:.4f}) | "
        f"fator_sato={melhor_so:.2f} (Dice treino={score_so:.4f})"
    )
    return melhor_cf, melhor_so


def executar_avaliacao_segmentacao(df_shg: pd.DataFrame) -> pd.DataFrame:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    registros = []

    for _, row in df_shg.iterrows():
        iid = str(row["image_id"])
        gt = carregar_mascara_binaria(Path(row["gt_path"])) if row["has_gt"] else None

        for metodo in METODOS_ORDEM:
            mask_p = caminho_mascara_metodo(metodo, iid)
            if not mask_p.exists():
                continue
            pred = carregar_mascara_binaria(mask_p)
            cob_pred = 100.0 * float((pred > 0).mean())

            reg = {
                "image_id": iid,
                "condicao": row["condicao"],
                "split": row["split"],
                "has_gt": bool(row["has_gt"]),
                "metodo": metodo,
                "metodo_label": METODOS_LABELS[metodo],
                "cobertura_pred_pct": cob_pred,
                "cobertura_gt_pct": np.nan,
                "erro_cobertura_abs_pp": np.nan,
                "dice": np.nan,
                "iou": np.nan,
                "precisao": np.nan,
                "revocacao": np.nan,
            }
            if gt is not None:
                m = calcular_metricas_binarias(pred, gt)
                reg.update(m)
            registros.append(reg)

    df_met = pd.DataFrame(registros)
    df_met.to_csv(METRICAS_IMG_CSV, index=False)
    print(f"[OK] Métricas de segmentação salvas em: {METRICAS_IMG_CSV}")
    return df_met


# ==============================================================================
# Etapa 3: Morfometria de fibras individuais (Comprimento, Largura, Orientação)
# ==============================================================================


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _ordenar_pixels_ramo(coords_yx: np.ndarray) -> np.ndarray:
    n = len(coords_yx)
    if n <= 2:
        return coords_yx

    pts = [tuple(map(int, p)) for p in coords_yx]
    pt_set = set(pts)
    adj: dict[tuple[int, int], list[tuple[int, int]]] = {p: [] for p in pts}
    for y, x in pts:
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                nb = (y + dy, x + dx)
                if nb in pt_set:
                    adj[(y, x)].append(nb)

    extremidades = [p for p, nbs in adj.items() if len(nbs) == 1]
    inicio = extremidades[0] if extremidades else pts[0]

    ordenado = [inicio]
    visitados = {inicio}
    atual = inicio
    while True:
        prox = [nb for nb in adj[atual] if nb not in visitados]
        if not prox:
            break
        prox.sort(key=lambda nb: abs(nb[0] - atual[0]) + abs(nb[1] - atual[1]))
        atual = prox[0]
        visitados.add(atual)
        ordenado.append(atual)

    return np.array(ordenado, dtype=np.int32)


def extrair_fibras_individuais(
    mask_bin: np.ndarray,
    img_raw: np.ndarray,
    min_comprimento_px: float = 8.0,
    max_angulo_juncao_deg: float = 35.0,
) -> list[dict[str, float]]:
    mask = mask_bin > 0
    if int(mask.sum()) < 5:
        return []

    edt = ndi.distance_transform_edt(mask)
    skel = skeletonize(mask)
    if int(skel.sum()) < 3:
        return []

    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    vizinhos = ndi.convolve(skel.astype(np.uint8), kernel, mode="constant", cval=0) * skel
    juncoes = vizinhos >= 3

    ramos_mask = skel & (~juncoes)
    lbl_ramos, n_ramos = ndi.label(ramos_mask, structure=np.ones((3, 3), dtype=int))
    if n_ramos == 0:
        return []

    ramos_ordenados: list[np.ndarray] = []
    for r_id in range(1, n_ramos + 1):
        coords = np.argwhere(lbl_ramos == r_id)
        ramos_ordenados.append(_ordenar_pixels_ramo(coords))

    uf = _UnionFind(n_ramos)
    lbl_juncoes, n_junc = ndi.label(juncoes, structure=np.ones((3, 3), dtype=int))
    cos_limite = -math.cos(math.radians(max_angulo_juncao_deg))

    if n_junc > 0:
        terminais = []
        for idx_r, coords in enumerate(ramos_ordenados):
            if len(coords) < 2:
                terminais.append((idx_r, 0, coords[0], np.zeros(2)))
                continue
            k = min(5, len(coords) - 1)
            v0 = coords[k].astype(np.float64) - coords[0].astype(np.float64)
            n0 = np.linalg.norm(v0)
            v0 = (v0 / n0) if n0 > 0 else np.zeros(2)
            terminais.append((idx_r, 0, coords[0], v0))

            v1 = coords[-1 - k].astype(np.float64) - coords[-1].astype(np.float64)
            n1 = np.linalg.norm(v1)
            v1 = (v1 / n1) if n1 > 0 else np.zeros(2)
            terminais.append((idx_r, 1, coords[-1], v1))

        for j_id in range(1, n_junc + 1):
            j_coords = np.argwhere(lbl_juncoes == j_id)
            incidentes = []
            for idx_t, (idx_r, ponta, pt_yx, vec) in enumerate(terminais):
                dist_cheb = np.max(np.abs(j_coords - pt_yx), axis=1).min()
                if dist_cheb <= 2 and np.linalg.norm(vec) > 0:
                    incidentes.append((idx_t, idx_r, ponta, vec))

            if len(incidentes) < 2:
                continue

            pares_candidatos = []
            for i in range(len(incidentes)):
                for j in range(i + 1, len(incidentes)):
                    ti, ri, _, vi = incidentes[i]
                    tj, rj, _, vj = incidentes[j]
                    if ri == rj:
                        continue
                    alinhamento = float(np.dot(vi, vj))
                    if alinhamento <= cos_limite:
                        pares_candidatos.append((alinhamento, ti, tj, ri, rj))

            pares_candidatos.sort(key=lambda x: x[0])
            usados = set()
            for _, ti, tj, ri, rj in pares_candidatos:
                if ti in usados or tj in usados:
                    continue
                uf.union(ri, rj)
                usados.add(ti)
                usados.add(tj)

    grupos: dict[int, list[np.ndarray]] = {}
    for idx_r, coords in enumerate(ramos_ordenados):
        raiz = uf.find(idx_r)
        grupos.setdefault(raiz, []).append(coords)

    fibras = []
    for fibra_idx, lista_coords in enumerate(grupos.values(), 1):
        pts = np.vstack(lista_coords)
        if len(lista_coords) > 1:
            pts = _ordenar_pixels_ramo(np.unique(pts, axis=0))

        if len(pts) < 2:
            continue

        diffs = np.diff(pts.astype(np.float64), axis=0)
        comprimento = float(np.hypot(diffs[:, 0], diffs[:, 1]).sum())
        if comprimento < min_comprimento_px:
            continue

        corda = float(np.hypot(pts[-1, 0] - pts[0, 0], pts[-1, 1] - pts[0, 1]))
        tortuosidade = (comprimento / corda) if corda > 1e-6 else 1.0

        larguras = 2.0 * edt[pts[:, 0], pts[:, 1]]
        largura_media = float(np.mean(larguras))
        largura_mediana = float(np.median(larguras))

        xy = np.column_stack([pts[:, 1], pts[:, 0]]).astype(np.float64)
        xy_cent = xy - xy.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(xy_cent, full_matrices=False)
        vx, vy = float(vt[0, 0]), float(vt[0, 1])
        orientacao_deg = float(math.degrees(math.atan2(vy, vx)) % 180.0)

        intensidade_media = float(np.mean(img_raw[pts[:, 0], pts[:, 1]]))

        fibras.append(
            {
                "fibra_id": fibra_idx,
                "comprimento_px": comprimento,
                "largura_media_px": largura_media,
                "largura_mediana_px": largura_mediana,
                "orientacao_deg": orientacao_deg,
                "tortuosidade": tortuosidade,
                "intensidade_shg_media": intensidade_media,
            }
        )

    return fibras


def executar_morfometria(
    df_shg: pd.DataFrame, min_comprimento_px: float = 8.0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    todas_fibras = []
    resumos_img = []

    for _, row in df_shg.iterrows():
        iid = str(row["image_id"])
        img_raw = load_2d(Path(str(row["raw_path"])))

        fontes: list[tuple[str, Path]] = []
        if row["has_gt"]:
            fontes.append(("ground_truth", Path(str(row["gt_path"]))))
        for metodo in METODOS_ORDEM:
            mp = caminho_mascara_metodo(metodo, iid)
            if mp.exists():
                fontes.append((metodo, mp))

        for metodo, mask_p in fontes:
            mask = carregar_mascara_binaria(mask_p)
            fibras = extrair_fibras_individuais(
                mask, img_raw, min_comprimento_px=min_comprimento_px
            )

            for f in fibras:
                todas_fibras.append(
                    {
                        "image_id": iid,
                        "condicao": row["condicao"],
                        "split": row["split"],
                        "has_gt": bool(row["has_gt"]),
                        "metodo": metodo,
                        **f,
                    }
                )

            if fibras:
                comps = np.array([f["comprimento_px"] for f in fibras], dtype=np.float64)
                largs = np.array([f["largura_media_px"] for f in fibras], dtype=np.float64)
                angs_rad = np.radians([f["orientacao_deg"] for f in fibras])
                torts = np.array([f["tortuosidade"] for f in fibras], dtype=np.float64)

                c2 = float(np.mean(np.cos(2.0 * angs_rad)))
                s2 = float(np.mean(np.sin(2.0 * angs_rad)))
                coerencia = float(math.hypot(c2, s2))
                ang_medio = float((0.5 * math.degrees(math.atan2(s2, c2))) % 180.0)

                resumos_img.append(
                    {
                        "image_id": iid,
                        "condicao": row["condicao"],
                        "split": row["split"],
                        "has_gt": bool(row["has_gt"]),
                        "metodo": metodo,
                        "n_fibras": len(fibras),
                        "comprimento_medio_px": float(np.mean(comps)),
                        "comprimento_mediano_px": float(np.median(comps)),
                        "largura_media_px": float(np.mean(largs)),
                        "largura_mediana_px": float(np.median(largs)),
                        "orientacao_media_deg": ang_medio,
                        "alinhamento_coerencia": coerencia,
                        "tortuosidade_media": float(np.mean(torts)),
                    }
                )
            else:
                resumos_img.append(
                    {
                        "image_id": iid,
                        "condicao": row["condicao"],
                        "split": row["split"],
                        "has_gt": bool(row["has_gt"]),
                        "metodo": metodo,
                        "n_fibras": 0,
                        "comprimento_medio_px": np.nan,
                        "comprimento_mediano_px": np.nan,
                        "largura_media_px": np.nan,
                        "largura_mediana_px": np.nan,
                        "orientacao_media_deg": np.nan,
                        "alinhamento_coerencia": np.nan,
                        "tortuosidade_media": np.nan,
                    }
                )

    df_fibras = pd.DataFrame(todas_fibras)
    df_resumo = pd.DataFrame(resumos_img)
    df_fibras.to_csv(MORFO_FIBRAS_CSV, index=False)
    df_resumo.to_csv(MORFO_IMG_CSV, index=False)
    print(f"[OK] Morfometria individual salva em: {MORFO_FIBRAS_CSV}")
    print(f"[OK] Resumo morfométrico salvo em   : {MORFO_IMG_CSV}")
    return df_fibras, df_resumo


# ==============================================================================
# Etapa 4: Tabela comparativa consolidada
# ==============================================================================


def gerar_tabela_comparativa(
    df_met: pd.DataFrame, df_morfo: pd.DataFrame
) -> pd.DataFrame:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df_full = pd.merge(
        df_met,
        df_morfo.drop(columns=["condicao", "split", "has_gt"], errors="ignore"),
        on=["image_id", "metodo"],
        how="left",
    )

    gt_ref = df_morfo[df_morfo["metodo"] == "ground_truth"][
        ["image_id", "n_fibras", "comprimento_medio_px", "largura_media_px"]
    ].rename(
        columns={
            "n_fibras": "gt_n_fibras",
            "comprimento_medio_px": "gt_comprimento_px",
            "largura_media_px": "gt_largura_px",
        }
    )
    df_full = pd.merge(df_full, gt_ref, on="image_id", how="left")

    df_full["erro_abs_contagem_fibras"] = (
        df_full["n_fibras"] - df_full["gt_n_fibras"]
    ).abs()
    df_full["erro_rel_comprimento_pct"] = (
        100.0
        * (df_full["comprimento_medio_px"] - df_full["gt_comprimento_px"]).abs()
        / df_full["gt_comprimento_px"].replace(0, np.nan)
    )
    df_full["erro_rel_largura_pct"] = (
        100.0
        * (df_full["largura_media_px"] - df_full["gt_largura_px"]).abs()
        / df_full["gt_largura_px"].replace(0, np.nan)
    )

    subconjuntos = [
        ("test_gt", df_full[(df_full["split"] == "test") & (df_full["has_gt"])]),
        ("train_gt", df_full[(df_full["split"] == "train") & (df_full["has_gt"])]),
        ("todas_gt", df_full[df_full["has_gt"]]),
        ("todas_imagens_35", df_full),
    ]

    linhas = []
    for nome_sub, sub in subconjuntos:
        if sub.empty:
            continue
        condicoes = ["TODAS"] + sorted(sub["condicao"].dropna().unique().tolist())
        for cond in condicoes:
            bloco = sub if cond == "TODAS" else sub[sub["condicao"] == cond]
            if bloco.empty:
                continue
            for metodo in METODOS_ORDEM:
                m_df = bloco[bloco["metodo"] == metodo]
                if m_df.empty:
                    continue
                linhas.append(
                    {
                        "subconjunto": nome_sub,
                        "condicao": cond,
                        "metodo": metodo,
                        "metodo_label": METODOS_LABELS[metodo],
                        "n_imagens": int(len(m_df)),
                        "dice_medio": float(m_df["dice"].mean()),
                        "dice_dp": float(m_df["dice"].std(ddof=1)) if len(m_df) > 1 else 0.0,
                        "iou_medio": float(m_df["iou"].mean()),
                        "iou_dp": float(m_df["iou"].std(ddof=1)) if len(m_df) > 1 else 0.0,
                        "precisao_media": float(m_df["precisao"].mean()),
                        "revocacao_media": float(m_df["revocacao"].mean()),
                        "cobertura_pred_pct": float(m_df["cobertura_pred_pct"].mean()),
                        "cobertura_gt_pct": float(m_df["cobertura_gt_pct"].mean()),
                        "n_fibras_medio": float(m_df["n_fibras"].mean()),
                        "comprimento_medio_px": float(m_df["comprimento_medio_px"].mean()),
                        "largura_media_px": float(m_df["largura_media_px"].mean()),
                        "coerencia_orientacao": float(m_df["alinhamento_coerencia"].mean()),
                        "tortuosidade_media": float(m_df["tortuosidade_media"].mean()),
                        "erro_abs_contagem_fibras": float(m_df["erro_abs_contagem_fibras"].mean()),
                        "erro_rel_comprimento_pct": float(m_df["erro_rel_comprimento_pct"].mean()),
                        "erro_rel_largura_pct": float(m_df["erro_rel_largura_pct"].mean()),
                    }
                )

    df_comp = pd.DataFrame(linhas)
    df_comp.to_csv(TABELA_COMP_CSV, index=False)
    print(f"[OK] Tabela comparativa salva em    : {TABELA_COMP_CSV}")

    visao_teste = df_comp[
        (df_comp["subconjunto"] == "test_gt") & (df_comp["condicao"] == "TODAS")
    ]
    if not visao_teste.empty:
        print("\n=== Resumo Comparativo — Conjunto de Teste (Sem Vazamento) ===")
        cols_print = [
            "metodo_label",
            "n_imagens",
            "dice_medio",
            "iou_medio",
            "cobertura_pred_pct",
            "comprimento_medio_px",
            "largura_media_px",
        ]
        print(visao_teste[cols_print].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    return df_comp


# ==============================================================================
# Etapa 5: Overlays revisáveis e figura-síntese
# ==============================================================================


def _renderizar_painel_overlay(
    ax,
    img_norm: np.ndarray,
    pred: np.ndarray | None,
    gt: np.ndarray | None,
    titulo: str,
) -> None:
    base_rgb = np.stack([img_norm, img_norm, img_norm], axis=-1)
    if pred is None:
        ax.imshow(base_rgb)
        ax.set_title(f"{titulo}\n[Não executado]", fontsize=8.5)
        ax.axis("off")
        return

    overlay = base_rgb.copy()
    p = pred > 0
    if gt is not None:
        g = gt > 0
        tp = p & g
        fp = p & (~g)
        fn = (~p) & g
        overlay[tp] = 0.45 * overlay[tp] + 0.55 * np.array([0.0, 0.95, 0.25])
        overlay[fp] = 0.45 * overlay[fp] + 0.55 * np.array([0.95, 0.15, 0.85])
        overlay[fn] = 0.45 * overlay[fn] + 0.55 * np.array([1.0, 0.85, 0.0])
    else:
        overlay[p] = 0.45 * overlay[p] + 0.55 * np.array([0.0, 0.9, 0.35])
        skel = skeletonize(p)
        overlay[skel] = np.array([1.0, 0.25, 0.1])

    ax.imshow(np.clip(overlay, 0.0, 1.0))
    ax.set_title(titulo, fontsize=8.5)
    ax.axis("off")


def gerar_overlays_revisaveis(
    df_shg: pd.DataFrame,
    df_met: pd.DataFrame,
    df_morfo: pd.DataFrame,
    todos: bool = False,
) -> None:
    OVERLAYS_DIR.mkdir(parents=True, exist_ok=True)
    alvo = df_shg if todos else df_shg[df_shg["has_gt"]].copy()

    met_idx = df_met.set_index(["image_id", "metodo"]).to_dict("index")
    morfo_idx = df_morfo.set_index(["image_id", "metodo"]).to_dict("index")

    gerados = 0
    for _, row in alvo.iterrows():
        iid = str(row["image_id"])
        cond = str(row["condicao"]).replace(" ", "_")
        out_png = OVERLAYS_DIR / f"{cond}_{row['split']}_{iid}_comparativo.png"

        img_norm = norm_percentile(load_2d(Path(str(row["raw_path"]))), 1.0, 99.5)
        gt = carregar_mascara_binaria(Path(str(row["gt_path"]))) if row["has_gt"] else None

        fig, axes = plt.subplots(1, 5, figsize=(22, 5.2))
        fig.suptitle(
            f"SHG BOCA | ID: {iid} | Condição: {row['condicao']} | Split: {row['split']} "
            + ("(Verde=TP, Magenta=FP, Amarelo=FN)" if gt is not None else "(Verde=Máscara, Vermelho=Esqueleto)"),
            fontsize=10.5,
            fontweight="bold",
        )

        p0 = np.stack([img_norm, img_norm, img_norm], axis=-1)
        if gt is not None:
            borda_gt = find_boundaries(gt > 0, mode="outer")
            p0[borda_gt] = np.array([0.0, 0.9, 1.0])
            gt_m = morfo_idx.get((iid, "ground_truth"), {})
            t0 = (
                f"SHG + Padrão-Ouro (Ciano)\n"
                f"Cov: {100.0 * (gt > 0).mean():.1f}% | Fibras: {gt_m.get('n_fibras', 0)}"
            )
        else:
            t0 = "SHG Original (Sem Gabarito)"
        axes[0].imshow(np.clip(p0, 0.0, 1.0))
        axes[0].set_title(t0, fontsize=8.5)
        axes[0].axis("off")

        for idx_m, metodo in enumerate(METODOS_ORDEM, 1):
            mask_p = caminho_mascara_metodo(metodo, iid)
            pred = carregar_mascara_binaria(mask_p) if mask_p.exists() else None
            m_info = met_idx.get((iid, metodo), {})
            f_info = morfo_idx.get((iid, metodo), {})

            if pred is not None and gt is not None:
                titulo = (
                    f"{METODOS_LABELS[metodo]}\n"
                    f"Dice={m_info.get('dice', 0):.3f} | IoU={m_info.get('iou', 0):.3f} | "
                    f"L={f_info.get('comprimento_medio_px', 0):.1f}px"
                )
            elif pred is not None:
                titulo = (
                    f"{METODOS_LABELS[metodo]}\n"
                    f"Cov={m_info.get('cobertura_pred_pct', 0):.1f}% | "
                    f"Fibras={f_info.get('n_fibras', 0)} | L={f_info.get('comprimento_medio_px', 0):.1f}px"
                )
            else:
                titulo = METODOS_LABELS[metodo]

            _renderizar_painel_overlay(axes[idx_m], img_norm, pred, gt, titulo)

        plt.tight_layout()
        fig.savefig(out_png, dpi=180)
        plt.close(fig)
        gerados += 1

    gt_rows = df_met[df_met["has_gt"]].copy()
    if not gt_rows.empty:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        metodos_presentes = [m for m in METODOS_ORDEM if m in gt_rows["metodo"].unique()]
        labels = [METODOS_LABELS[m] for m in metodos_presentes]

        for ax, metrica, nome_y in [
            (axes[0], "dice", "Dice (Padrão-Ouro SHG)"),
            (axes[1], "iou", "IoU (Padrão-Ouro SHG)"),
        ]:
            dados = [
                gt_rows[gt_rows["metodo"] == m][metrica].dropna().values
                for m in metodos_presentes
            ]
            ax.boxplot(dados, tick_labels=labels, patch_artist=True)
            ax.set_ylabel(nome_y)
            ax.set_ylim(0.0, 1.05)
            ax.grid(axis="y", linestyle="--", alpha=0.4)
            ax.tick_params(axis="x", rotation=15)

        plt.tight_layout()
        fig.savefig(FIGURA_RESUMO_PNG, dpi=200)
        plt.close(fig)

    print(f"[OK] {gerados} overlay(s) revisável(is) salvo(s) em: {OVERLAYS_DIR}")


# ==============================================================================
# Checklist / Status
# ==============================================================================


def cmd_status(_: argparse.Namespace | None = None) -> None:
    df_shg = carregar_dataset_shg()
    n_total = len(df_shg)
    n_gt = int(df_shg["has_gt"].sum())

    print("\n=== Checklist de Conclusão — Atividade 2 (Colágeno SHG / BOCA) ===")
    print(f"Total de imagens SHG: {n_total} | Com padrão-ouro: {n_gt}")
    for metodo in METODOS_ORDEM:
        n_ok = sum(
            caminho_mascara_metodo(metodo, str(iid)).exists()
            for iid in df_shg["image_id"]
        )
        tag = "OK" if n_ok == n_total else ("PARCIAL" if n_ok > 0 else "PENDENTE")
        print(f"[{tag}] Máscaras {METODOS_LABELS[metodo]}: {n_ok}/{n_total}")

    n_ov = len(list(OVERLAYS_DIR.glob("*.png"))) if OVERLAYS_DIR.exists() else 0
    print(f"[{'OK' if METRICAS_IMG_CSV.exists() else 'PENDENTE'}] Métricas Dice/IoU/Cobertura: {METRICAS_IMG_CSV.name}")
    print(f"[{'OK' if MORFO_FIBRAS_CSV.exists() else 'PENDENTE'}] Morfometria de fibras: {MORFO_FIBRAS_CSV.name}")
    print(f"[{'OK' if TABELA_COMP_CSV.exists() else 'PENDENTE'}] Tabela comparativa: {TABELA_COMP_CSV.name}")
    print(f"[{'OK' if n_ov >= n_gt else 'PENDENTE'}] Overlays revisáveis: {n_ov} arquivo(s) em {OVERLAYS_DIR}")


# ==============================================================================
# Orquestrador sequencial completo
# ==============================================================================


def cmd_pipeline(args: argparse.Namespace) -> None:
    print("Iniciando pipeline completo da Atividade 2 (Colágeno SHG - BOCA)...\n")
    df_shg = carregar_dataset_shg()

    fator_cf, fator_so = (1.0, 1.0)
    if args.calibrar_treino:
        fator_cf, fator_so = calibrar_fatores_no_treino(df_shg)

    executar_segmentacoes(
        df_shg=df_shg,
        cellpose_python=args.cellpose_python,
        ctfire_dir=Path(args.ctfire_dir) if args.ctfire_dir else None,
        fator_curvelet=fator_cf,
        fator_sato=fator_so,
        pular_cellpose=args.pular_cellpose,
        cellpose_limit=args.cellpose_limit,
        cellpose_max_dim=args.cellpose_max_dim,
        force=args.force or args.calibrar_treino,
    )

    df_met = executar_avaliacao_segmentacao(df_shg)
    _, df_morfo = executar_morfometria(df_shg, min_comprimento_px=args.min_comprimento)
    gerar_tabela_comparativa(df_met, df_morfo)
    gerar_overlays_revisaveis(df_shg, df_met, df_morfo, todos=args.todos)
    cmd_status()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pipeline consolidado da Atividade 2 — Colágeno SHG (BOCA)."
    )
    parser.add_argument("--cellpose-python", default=None)
    parser.add_argument("--ctfire-dir", default=None)
    parser.add_argument("--pular-cellpose", action="store_true")
    parser.add_argument(
        "--cellpose-limit",
        type=int,
        default=None,
        help="Processa no máximo N imagens pendentes no Cellpose-SAM por execução.",
    )
    parser.add_argument(
        "--cellpose-max-dim",
        type=int,
        default=512,
        help="Lado máximo em pixels para inferência do Cellpose-SAM em CPU (0 = nativa).",
    )
    parser.add_argument("--calibrar-treino", action="store_true")
    parser.add_argument("--min-comprimento", type=float, default=8.0)
    parser.add_argument("--todos", action="store_true")
    parser.add_argument("--force", action="store_true")

    sub = parser.add_subparsers(dest="comando")

    p_status = sub.add_parser("status", help="Exibe o checklist da Atividade 2.")
    p_status.set_defaults(func=cmd_status)

    p_seg = sub.add_parser("segmentar", help="Roda apenas as segmentações faltantes.")
    p_seg.add_argument("--cellpose-python", default=None)
    p_seg.add_argument("--ctfire-dir", default=None)
    p_seg.add_argument("--pular-cellpose", action="store_true")
    p_seg.add_argument("--cellpose-limit", type=int, default=None)
    p_seg.add_argument("--cellpose-max-dim", type=int, default=512)
    p_seg.add_argument("--force", action="store_true")
    p_seg.set_defaults(
        func=lambda a: executar_segmentacoes(
            carregar_dataset_shg(),
            cellpose_python=a.cellpose_python,
            ctfire_dir=Path(a.ctfire_dir) if a.ctfire_dir else None,
            pular_cellpose=a.pular_cellpose,
            cellpose_limit=a.cellpose_limit,
            cellpose_max_dim=a.cellpose_max_dim,
            force=a.force,
        )
    )

    p_av = sub.add_parser("avaliar", help="Recalcula métricas de segmentação e tabela comparativa.")
    p_av.set_defaults(
        func=lambda _: gerar_tabela_comparativa(
            executar_avaliacao_segmentacao(carregar_dataset_shg()),
            pd.read_csv(MORFO_IMG_CSV)
            if MORFO_IMG_CSV.exists()
            else executar_morfometria(carregar_dataset_shg())[1],
        )
    )

    p_morfo = sub.add_parser("morfometria", help="Extrai morfometria individual das fibras.")
    p_morfo.add_argument("--min-comprimento", type=float, default=8.0)
    p_morfo.set_defaults(
        func=lambda a: executar_morfometria(
            carregar_dataset_shg(), min_comprimento_px=a.min_comprimento
        )
    )

    p_ov = sub.add_parser("overlays", help="Gera os overlays comparativos revisáveis.")
    p_ov.add_argument("--todos", action="store_true")
    p_ov.set_defaults(
        func=lambda a: gerar_overlays_revisaveis(
            carregar_dataset_shg(),
            pd.read_csv(METRICAS_IMG_CSV),
            pd.read_csv(MORFO_IMG_CSV),
            todos=a.todos,
        )
    )

    args = parser.parse_args()
    if not hasattr(args, "func"):
        cmd_pipeline(args)
        return
    args.func(args)


if __name__ == "__main__":
    main()
