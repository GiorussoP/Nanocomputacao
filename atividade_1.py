#!/usr/bin/env python3
"""
Atividade 1 — Pipeline completo e idempotente (BOCA).

Execução padrão (roda todas as etapas em ordem e pula o que já está pronto):
  python3 atividade_1.py

Subcomandos opcionais para execução isolada:
  python3 atividade_1.py status
  python3 atividade_1.py revisar [--pair-id ID] [--revisar-registrados]
  python3 atividade_1.py registrar [--pair-id ID] [--mode auto|landmarks|identity] [--forcar-exportacao]
  python3 atividade_1.py anotar [--max-pares 12] [--pair-id ID]
  python3 atividade_1.py finalizar [--responsavel "Gabriel"] [--force]
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.filters import sobel
from skimage.registration import phase_cross_correlation
from skimage.transform import EuclideanTransform, SimilarityTransform, warp
import tifffile


ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "raw" / "IMAGENS"
META_DIR = ROOT / "metadata"
ANN_DIR = ROOT / "annotations" / "BOCA"
SEG_DIR = ROOT / "processed" / "foundation_microscopy" / "segmentations" / "BOCA"
REGISTERED_DIR = ROOT / "processed" / "foundation_microscopy" / "registered" / "BOCA"

IMAGES_CSV = META_DIR / "foundation_images.csv"
MANIFESTO_CSV = META_DIR / "manifesto_pareamento_boca.csv"
PMT_JSON = META_DIR / "confirmacao_detectores.json"
SPLIT_IMAGES_CSV = META_DIR / "split_boca.csv"
SPLIT_CAMPOS_CSV = META_DIR / "split_boca_campos.csv"

MANIFESTO_COLS = [
    "pair_id",
    "condicao",
    "shg_id",
    "tpef_id",
    "shg_relpath",
    "tpef_relpath",
    "mesmo_campo",
    "alinhado",
    "observacao",
]


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
    img: np.ndarray,
    p_low: float = 1.0,
    p_high: float = 99.0,
    limits: tuple[float, float] | None = None,
) -> np.ndarray:
    arr = img.astype(np.float32)
    if limits is None:
        lo, hi = np.percentile(arr, (p_low, p_high))
    else:
        lo, hi = limits
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


def resolver_raw_path(row: pd.Series, rel_col: str, path_col: str) -> Path:
    rel_candidate = RAW_DIR / str(row[rel_col])
    if rel_candidate.exists():
        return rel_candidate
    return Path(str(row[path_col]))


def resolver_caminhos_par(
    pair_id: str, shg_id: str, tpef_id: str, shg_raw: Path, tpef_raw: Path
) -> tuple[Path, Path, bool]:
    pair_dir = REGISTERED_DIR / str(pair_id).replace(":", "_")
    shg_reg = pair_dir / f"{shg_id}_SHG_ref.tif"
    tpef_reg = pair_dir / f"{tpef_id}_TPEF_reg.tif"
    if tpef_reg.exists():
        return (shg_reg if shg_reg.exists() else shg_raw), tpef_reg, True
    return shg_raw, tpef_raw, False


def detectar_anotacoes() -> dict[str, set[str]]:
    encontradas: dict[str, set[str]] = {"SHG": set(), "TPEF": set()}
    for mod in ("SHG", "TPEF"):
        mod_dir = ANN_DIR / mod
        if not mod_dir.exists():
            continue
        for tif in mod_dir.glob("*.tif"):
            if tif.name.endswith("_unaligned_backup.tif"):
                continue
            encontradas[mod].add(tif.stem)
    return encontradas


def listar_grupos_candidatos_boca() -> list[tuple[str, pd.DataFrame]]:
    df = pd.read_csv(IMAGES_CSV)
    boca = df[df["group"] == "BOCA"].copy()

    pair_col = None
    for col in boca.columns:
        if boca[col].astype(str).str.startswith("candidate:").any():
            pair_col = col
            break
    if pair_col is None:
        raise ValueError("Nenhuma coluna com 'candidate:' encontrada em foundation_images.csv.")

    candidatos = boca[boca[pair_col].astype(str).str.startswith("candidate:")].copy()
    validos: list[tuple[str, pd.DataFrame]] = []
    for pair_id, grp in candidatos.groupby(pair_col):
        if not grp[grp["modality"] == "SHG"].empty and not grp[grp["modality"] == "TPEF"].empty:
            validos.append((str(pair_id), grp))
    return validos


# ==============================================================================
# Etapa 1: Revisão visual dos pares SHG / TPEF
# ==============================================================================


def mostrar_par(shg_path: Path, tpef_path: Path, titulo: str) -> None:
    img_shg = norm_percentile(load_2d(shg_path), 1.0, 99.5)
    img_tpef = norm_percentile(load_2d(tpef_path), 1.0, 99.5)

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle(titulo, fontsize=11, fontweight="bold")

    axes[0].imshow(img_shg, cmap="Greens")
    axes[0].set_title(f"SHG (Verde)\n{shg_path.name[:35]}...", fontsize=9)

    axes[1].imshow(img_tpef, cmap="RdPu")
    axes[1].set_title(f"TPEF (Magenta)\n{tpef_path.name[:35]}...", fontsize=9)

    if img_shg.shape == img_tpef.shape:
        rgb = np.stack([img_tpef, img_shg, img_tpef], axis=-1)
        axes[2].imshow(rgb)
        axes[2].set_title("Sobreposição (Branco = Alinhado)", fontsize=9)
    else:
        axes[2].text(
            0.5,
            0.5,
            f"Shapes diferentes!\nSHG: {img_shg.shape}\nTPEF: {img_tpef.shape}",
            ha="center",
            va="center",
        )
        axes[2].set_title("Incompatível", fontsize=9)

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def salvar_registro_manifesto(out_path: Path, registro: dict) -> None:
    if out_path.exists():
        df_out = pd.read_csv(out_path, dtype={"observacao": "string"})
    else:
        df_out = pd.DataFrame(columns=MANIFESTO_COLS)

    mask = df_out["pair_id"].astype(str) == str(registro["pair_id"])
    if mask.any():
        for col, val in registro.items():
            df_out.loc[mask, col] = val
    else:
        df_out = pd.concat([df_out, pd.DataFrame([registro])], ignore_index=True)

    df_out.to_csv(out_path, index=False)


def executar_revisar(pair_id_filtro: str | None = None, revisar_registrados: bool = False) -> bool:
    """Retorna True se todos os pares candidatos foram revisados; False se interrompido pelo usuário."""
    grupos = listar_grupos_candidatos_boca()

    prev_map: dict[str, dict] = {}
    if MANIFESTO_CSV.exists():
        prev = pd.read_csv(MANIFESTO_CSV, dtype={"observacao": "string"})
        prev_map = {str(r["pair_id"]): r.to_dict() for _, r in prev.iterrows()}

    if not pair_id_filtro and not revisar_registrados:
        pendentes = [pid for pid, _ in grupos if pid not in prev_map]
        if not pendentes:
            print(f"[PULAR] Etapa 1 (Revisão de pares): {len(grupos)}/{len(grupos)} pares já constam no manifesto.")
            return True
        print(f"[ETAPA 1] Iniciando revisão de {len(pendentes)} pares pendentes...")

    for idx, (pair_id_str, grp) in enumerate(grupos, 1):
        shg = grp[grp["modality"] == "SHG"].iloc[0]
        tpef = grp[grp["modality"] == "TPEF"].iloc[0]

        path_col = "path" if "path" in grp.columns else grp.columns[1]
        rel_col = "relative_path" if "relative_path" in grp.columns else grp.columns[2]
        id_col = grp.columns[0]

        shg_id = str(shg[id_col])
        tpef_id = str(tpef[id_col])
        shg_raw = resolver_raw_path(shg, rel_col, path_col)
        tpef_raw = resolver_raw_path(tpef, rel_col, path_col)

        shg_path, tpef_path, usa_registrado = resolver_caminhos_par(
            pair_id_str, shg_id, tpef_id, shg_raw, tpef_raw
        )

        if pair_id_filtro and pair_id_str != pair_id_filtro:
            continue
        if revisar_registrados and not usa_registrado:
            continue
        if not pair_id_filtro and not revisar_registrados and pair_id_str in prev_map:
            continue

        condicao = inferir_condicao(str(shg[rel_col]))
        tag = " [REGISTRADO]" if usa_registrado else ""
        obs_anterior = ""
        if pair_id_str in prev_map and pd.notna(prev_map[pair_id_str].get("observacao")):
            obs_anterior = str(prev_map[pair_id_str]["observacao"])

        print("=" * 70)
        print(f"[{idx}/{len(grupos)}] Par: {pair_id_str}{tag} | Condição: {condicao}")
        print(f"  SHG : {shg_path}")
        print(f"  TPEF: {tpef_path}")
        if obs_anterior:
            print(f"  Obs. atual: {obs_anterior}")
        print("Feche a janela da imagem para responder no terminal...")

        mostrar_par(shg_path, tpef_path, f"[{idx}/{len(grupos)}] {condicao} - {pair_id_str}{tag}")

        resp = input(
            "É o mesmo campo de visão? [s = sim alinhado | d = sim, mas deslocado | n = não/duvidoso | q = sair]: "
        ).strip().lower()
        if resp == "q":
            print("Revisão interrompida pelo usuário. Progresso salvo.")
            return False

        obs_input = input("Observação rápida (Enter mantém a atual): ").strip()
        obs_final = obs_input if obs_input else obs_anterior

        mesmo_campo = "sim" if resp in ("s", "d") else "nao"
        alinhado = "sim" if resp == "s" else ("deslocado" if resp == "d" else "nao")

        salvar_registro_manifesto(
            MANIFESTO_CSV,
            {
                "pair_id": pair_id_str,
                "condicao": condicao,
                "shg_id": shg_id,
                "tpef_id": tpef_id,
                "shg_relpath": str(shg[rel_col]),
                "tpef_relpath": str(tpef[rel_col]),
                "mesmo_campo": mesmo_campo,
                "alinhado": alinhado,
                "observacao": obs_final,
            },
        )
        print("Salvo!\n")

    return True


def cmd_revisar(args: argparse.Namespace) -> None:
    executar_revisar(pair_id_filtro=args.pair_id, revisar_registrados=args.revisar_registrados)


# ==============================================================================
# Etapa 2: Registro espacial SHG/TPEF e exportação dos pares alinhados
# ==============================================================================


def estimate_auto_transform(shg: np.ndarray, tpef: np.ndarray) -> tuple[SimilarityTransform, str]:
    shg_n = norm_percentile(shg, 1.0, 99.0)
    tpef_n = norm_percentile(tpef, 1.0, 99.0)
    shift, _, _ = phase_cross_correlation(sobel(shg_n), sobel(tpef_n), upsample_factor=10)
    dy, dx = float(shift[0]), float(shift[1])
    inv_tform = SimilarityTransform(translation=(-dx, -dy))
    return inv_tform, f"auto_shift(dy={dy:.2f}, dx={dx:.2f})"


def estimate_landmark_transform(
    shg: np.ndarray, tpef: np.ndarray, n_pts: int = 3
) -> tuple[EuclideanTransform, str]:
    shg_n = norm_percentile(shg, 1.0, 99.0)
    tpef_n = norm_percentile(tpef, 1.0, 99.0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    axes[0].imshow(shg_n, cmap="magma")
    axes[0].set_title(f"1. Referência: SHG (clique {n_pts} pontos)")
    axes[1].imshow(tpef_n, cmap="viridis")
    axes[1].set_title(f"2. Móvel: TPEF (clique os mesmos {n_pts} pontos na mesma ordem)")
    plt.tight_layout()

    pts_shg = np.array(plt.ginput(n_pts, timeout=0))
    pts_tpef = np.array(plt.ginput(n_pts, timeout=0))
    plt.close(fig)

    if len(pts_shg) < n_pts or len(pts_tpef) < n_pts:
        raise RuntimeError("Seleção de pontos incompleta.")

    inv_tform = EuclideanTransform()
    inv_tform.estimate(pts_shg, pts_tpef)
    dx, dy = inv_tform.translation
    ang_deg = np.degrees(inv_tform.rotation)
    return inv_tform, f"landmarks_rigid(dy={-dy:.2f}, dx={-dx:.2f}, rot={-ang_deg:.2f}deg)"


def save_overlay(shg: np.ndarray, tpef_before: np.ndarray, tpef_after: np.ndarray, out_png: Path) -> None:
    s = norm_percentile(shg, 1.0, 99.0)
    tpef_limits = tuple(np.percentile(tpef_before, (1.0, 99.0)))
    b = norm_percentile(tpef_before, limits=tpef_limits)
    a = norm_percentile(tpef_after, limits=tpef_limits)

    rgb_before = np.stack([s, b, s], axis=-1)  # SHG = magenta, TPEF = verde
    rgb_after = np.stack([s, a, s], axis=-1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(rgb_before)
    axes[0].set_title("Antes (SHG=Magenta, TPEF=Verde)")
    axes[0].axis("off")
    axes[1].imshow(rgb_after)
    axes[1].set_title("Depois do Registro")
    axes[1].axis("off")
    plt.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def sincronizar_mascaras_no_registro(pair_dir: Path, shg_id: str, tpef_id: str) -> None:
    shg_mask_path = ANN_DIR / "SHG" / f"{shg_id}.tif"
    shg_mask_out = pair_dir / f"{shg_id}_SHG_mask_ref.tif"
    if shg_mask_path.exists() and not shg_mask_out.exists():
        tifffile.imwrite(shg_mask_out, load_2d(shg_mask_path))

    tpef_mask_path = ANN_DIR / "TPEF" / f"{tpef_id}.tif"
    tpef_mask_out = pair_dir / f"{tpef_id}_TPEF_mask_reg.tif"
    if tpef_mask_path.exists() and not tpef_mask_out.exists():
        tifffile.imwrite(tpef_mask_out, load_2d(tpef_mask_path))


def process_pair_registration(row: pd.Series, mode: str, update_manifest: bool = True) -> str:
    pair_id = str(row["pair_id"])
    shg_id = str(row["shg_id"])
    tpef_id = str(row["tpef_id"])

    shg = load_2d(RAW_DIR / str(row["shg_relpath"]))
    tpef = load_2d(RAW_DIR / str(row["tpef_relpath"]))

    if mode == "identity":
        inv_tform = SimilarityTransform(translation=(0.0, 0.0))
        desc = "identity"
    elif mode == "auto":
        inv_tform, desc = estimate_auto_transform(shg, tpef)
    elif mode == "landmarks":
        inv_tform, desc = estimate_landmark_transform(shg, tpef, n_pts=3)
    else:
        raise ValueError(f"Modo inválido: {mode}")

    bg_cval = float(np.percentile(tpef, 5.0))
    tpef_reg = warp(
        tpef,
        inv_tform,
        output_shape=shg.shape,
        preserve_range=True,
        mode="constant",
        cval=bg_cval,
    ).astype(tpef.dtype)

    valid_fov = warp(
        np.ones(tpef.shape, dtype=np.uint8),
        inv_tform,
        output_shape=shg.shape,
        order=0,
        preserve_range=True,
        mode="constant",
        cval=0,
    ).astype(np.uint8)

    pair_out = REGISTERED_DIR / pair_id.replace(":", "_")
    pair_out.mkdir(parents=True, exist_ok=True)

    tifffile.imwrite(pair_out / f"{shg_id}_SHG_ref.tif", shg)
    tifffile.imwrite(pair_out / f"{tpef_id}_TPEF_reg.tif", tpef_reg)
    tifffile.imwrite(pair_out / "valid_fov_mask.tif", valid_fov)
    save_overlay(shg, tpef, tpef_reg, pair_out / "overlay_antes_depois.png")

    tpef_mask_path = ANN_DIR / "TPEF" / f"{tpef_id}.tif"
    backup_path = ANN_DIR / "TPEF" / f"{tpef_id}_unaligned_backup.tif"
    if tpef_mask_path.exists() or backup_path.exists():
        if not backup_path.exists():
            tifffile.imwrite(backup_path, load_2d(tpef_mask_path))
        tpef_mask = load_2d(backup_path)
        tpef_mask_reg = warp(
            tpef_mask,
            inv_tform,
            output_shape=shg.shape,
            order=0,
            preserve_range=True,
            mode="constant",
            cval=0,
        ).astype(tpef_mask.dtype)
        tifffile.imwrite(pair_out / f"{tpef_id}_TPEF_mask_reg.tif", tpef_mask_reg)
        tifffile.imwrite(tpef_mask_path, tpef_mask_reg)

    shg_mask_path = ANN_DIR / "SHG" / f"{shg_id}.tif"
    if shg_mask_path.exists():
        tifffile.imwrite(pair_out / f"{shg_id}_SHG_mask_ref.tif", load_2d(shg_mask_path))

    if update_manifest and mode != "identity":
        df = pd.read_csv(MANIFESTO_CSV, dtype={"observacao": "string"})
        idx = df.index[df["pair_id"] == pair_id]
        df.loc[idx, "alinhado"] = "sim"
        df.loc[idx, "observacao"] = desc
        df.to_csv(MANIFESTO_CSV, index=False)

    return desc


def executar_registrar_automatico(mode: str = "auto", forcar_exportacao: bool = False) -> None:
    if not MANIFESTO_CSV.exists():
        raise FileNotFoundError(f"Manifesto não encontrado: {MANIFESTO_CSV}")

    df = pd.read_csv(MANIFESTO_CSV, dtype={"observacao": "string"})
    deslocados = df[(df["mesmo_campo"] == "sim") & (df["alinhado"] == "deslocado")]

    if not deslocados.empty:
        print(f"[ETAPA 2a] Alinhando {len(deslocados)} par(es) deslocado(s)...")
        for _, row in deslocados.iterrows():
            desc = process_pair_registration(row, mode=mode, update_manifest=True)
            print(f"  -> [{row['pair_id']}] {desc}")
        df = pd.read_csv(MANIFESTO_CSV, dtype={"observacao": "string"})

    alinhados = df[df["alinhado"] == "sim"]
    exportados_agora = 0
    for _, row in alinhados.iterrows():
        pair_dir = REGISTERED_DIR / str(row["pair_id"]).replace(":", "_")
        ja_existe = (
            (pair_dir / f"{row['shg_id']}_SHG_ref.tif").exists()
            and (pair_dir / f"{row['tpef_id']}_TPEF_reg.tif").exists()
            and (pair_dir / "valid_fov_mask.tif").exists()
        )
        if ja_existe and not forcar_exportacao:
            sincronizar_mascaras_no_registro(pair_dir, str(row["shg_id"]), str(row["tpef_id"]))
            continue
        process_pair_registration(row, mode="identity", update_manifest=False)
        exportados_agora += 1

    if deslocados.empty and exportados_agora == 0:
        print(
            f"[PULAR] Etapa 2 (Registro espacial): {len(alinhados)}/{len(alinhados)} pares já alinhados e exportados."
        )
    else:
        print(f"[OK] Etapa 2 concluída: {len(alinhados)} pares disponíveis em {REGISTERED_DIR}")


def cmd_registrar(args: argparse.Namespace) -> None:
    if args.pair_id:
        df = pd.read_csv(MANIFESTO_CSV, dtype={"observacao": "string"})
        match = df[df["pair_id"] == args.pair_id]
        if match.empty:
            raise ValueError(f"pair_id não encontrado no manifesto: {args.pair_id}")
        desc = process_pair_registration(match.iloc[0], mode=args.mode, update_manifest=True)
        print(f"[OK] Concluído [{args.pair_id}]: {desc}")
        return

    executar_registrar_automatico(mode=args.mode, forcar_exportacao=args.forcar_exportacao)


# ==============================================================================
# Etapa 3: Editor interativo do gabarito manual (SHG e TPEF)
# ==============================================================================


class EditorMascara:
    def __init__(self, img: np.ndarray, mask_inicial: np.ndarray, titulo: str):
        self.img = img
        self.mask = (mask_inicial > 0).astype(np.uint8)
        self.raio = 8
        self.desenhando = False
        self.valor_pincel = 1
        self.salvo = False
        self.sair_tudo = False

        self.fig, (self.ax_orig, self.ax_edit) = plt.subplots(1, 2, figsize=(15, 7))
        self.fig.suptitle(
            f"{titulo}\n"
            "[Botão Esq: Pintar]  [Botão Dir: Apagar]  [Scroll/+/-: Tamanho Pincel]  "
            "[C: Limpar tudo]  [Enter: Salvar e Próxima]  [Q: Sair]",
            fontsize=10,
            fontweight="bold",
        )

        self.ax_orig.imshow(self.img, cmap="gray")
        self.ax_orig.set_title("Imagem Original (Referência)")
        self.ax_orig.axis("off")

        self.ax_edit.imshow(self.img, cmap="gray")
        self.overlay = self.ax_edit.imshow(self._criar_rgba(), interpolation="nearest")
        self.ax_edit.set_title(f"Máscara Editável (Raio pincel: {self.raio}px)")
        self.ax_edit.axis("off")

        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_move)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        plt.tight_layout()
        plt.show()

    def _criar_rgba(self) -> np.ndarray:
        rgba = np.zeros((*self.mask.shape, 4), dtype=np.float32)
        rgba[self.mask > 0] = [0.0, 1.0, 0.3, 0.45]
        return rgba

    def _aplicar_pincel(self, x: float | None, y: float | None) -> None:
        if x is None or y is None:
            return
        h, w = self.mask.shape
        cx, cy = int(round(x)), int(round(y))
        y_min, y_max = max(0, cy - self.raio), min(h, cy + self.raio + 1)
        x_min, x_max = max(0, cx - self.raio), min(w, cx + self.raio + 1)
        if y_min >= y_max or x_min >= x_max:
            return

        yy, xx = np.ogrid[y_min:y_max, x_min:x_max]
        disco = (xx - cx) ** 2 + (yy - cy) ** 2 <= self.raio ** 2
        self.mask[y_min:y_max, x_min:x_max][disco] = self.valor_pincel

        self.overlay.set_data(self._criar_rgba())
        self.fig.canvas.draw_idle()

    def _on_press(self, event) -> None:
        if event.inaxes != self.ax_edit:
            return
        if event.button == 1:
            self.desenhando = True
            self.valor_pincel = 1
            self._aplicar_pincel(event.xdata, event.ydata)
        elif event.button == 3:
            self.desenhando = True
            self.valor_pincel = 0
            self._aplicar_pincel(event.xdata, event.ydata)

    def _on_move(self, event) -> None:
        if self.desenhando and event.inaxes == self.ax_edit:
            self._aplicar_pincel(event.xdata, event.ydata)

    def _on_release(self, event) -> None:
        self.desenhando = False

    def _on_scroll(self, event) -> None:
        delta = 2 if event.button == "up" else -2
        self.raio = int(np.clip(self.raio + delta, 1, 50))
        self.ax_edit.set_title(f"Máscara Editável (Raio pincel: {self.raio}px)")
        self.fig.canvas.draw_idle()

    def _on_key(self, event) -> None:
        if event.key in ("+", "="):
            self.raio = min(50, self.raio + 2)
            self.ax_edit.set_title(f"Máscara Editável (Raio pincel: {self.raio}px)")
            self.fig.canvas.draw_idle()
        elif event.key == "-":
            self.raio = max(1, self.raio - 2)
            self.ax_edit.set_title(f"Máscara Editável (Raio pincel: {self.raio}px)")
            self.fig.canvas.draw_idle()
        elif event.key == "c":
            self.mask.fill(0)
            self.overlay.set_data(self._criar_rgba())
            self.fig.canvas.draw_idle()
        elif event.key == "enter":
            self.salvo = True
            plt.close(self.fig)
        elif event.key == "q":
            self.sair_tudo = True
            plt.close(self.fig)


def executar_anotar(max_pares: int = 12, pair_id_filtro: str | None = None) -> tuple[bool, int]:
    """Retorna (concluido_sem_interrupcao, qtd_novas_mascaras_salvas)."""
    manifesto = pd.read_csv(MANIFESTO_CSV)
    df_imgs = pd.read_csv(IMAGES_CSV)
    id_col = df_imgs.columns[0]
    path_col = "path" if "path" in df_imgs.columns else df_imgs.columns[1]
    rel_col = "relative_path" if "relative_path" in df_imgs.columns else df_imgs.columns[2]

    mapa_caminhos = {
        str(r[id_col]): resolver_raw_path(r, rel_col, path_col) for _, r in df_imgs.iterrows()
    }

    alinhados = manifesto[manifesto["alinhado"] == "sim"].copy()
    if pair_id_filtro:
        selecionados = alinhados[alinhados["pair_id"].astype(str) == pair_id_filtro]
    else:
        selecionados = alinhados.head(max_pares)

    tarefas_pendentes = []
    for idx, row in enumerate(selecionados.itertuples(), 1):
        for modalidade, img_id in [("SHG", str(row.shg_id)), ("TPEF", str(row.tpef_id))]:
            destino = ANN_DIR / modalidade / f"{img_id}.tif"
            if not destino.exists():
                tarefas_pendentes.append((idx, row, modalidade, img_id, destino))

    if not tarefas_pendentes:
        anot = detectar_anotacoes()
        print(
            f"[PULAR] Etapa 3 (Anotação do gabarito): máscaras alvo já concluídas "
            f"({len(anot['SHG'])} SHG | {len(anot['TPEF'])} TPEF)."
        )
        return True, 0

    print(f"[ETAPA 3] Iniciando anotação de {len(tarefas_pendentes)} máscara(s) pendente(s)...")
    print("  - Clique/Arraste ESQUERDO : Pintar estrutura (fibra)")
    print("  - Clique/Arraste DIREITO  : Apagar falso positivo (fundo)")
    print("  - Scroll ou teclas +/-    : Aumentar/diminuir pincel")
    print("  - Tecla C                 : Limpar máscara inteira")
    print("  - Tecla ENTER             : Salvar máscara e ir para a próxima")
    print("  - Tecla Q                 : Sair\n")

    salvas_agora = 0
    for idx, row, modalidade, img_id, destino in tarefas_pendentes:
        shg_reg, tpef_reg, _ = resolver_caminhos_par(
            str(row.pair_id),
            str(row.shg_id),
            str(row.tpef_id),
            mapa_caminhos[str(row.shg_id)],
            mapa_caminhos[str(row.tpef_id)],
        )
        img_path = shg_reg if modalidade == "SHG" else tpef_reg
        pred_path = SEG_DIR / modalidade / f"{img_id}.tif"

        img = norm_percentile(load_2d(img_path), 1.0, 99.5)
        mask_init = load_2d(pred_path) if pred_path.exists() else np.zeros_like(img, dtype=np.uint8)

        titulo = f"Campo {idx}/{len(selecionados)} ({row.condicao}) | {modalidade} | ID: {img_id}"
        editor = EditorMascara(img, mask_init, titulo)

        if editor.sair_tudo:
            print("Anotação interrompida pelo usuário.")
            return False, salvas_agora

        if editor.salvo:
            destino.parent.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(destino, editor.mask.astype(np.uint8))
            pair_dir = REGISTERED_DIR / str(row.pair_id).replace(":", "_")
            if pair_dir.exists():
                sincronizar_mascaras_no_registro(pair_dir, str(row.shg_id), str(row.tpef_id))
            salvas_agora += 1
            print(f"  -> Salvo em: {destino}")

    return True, salvas_agora


def cmd_anotar(args: argparse.Namespace) -> None:
    executar_anotar(max_pares=args.max_pares, pair_id_filtro=args.pair_id)


# ==============================================================================
# Etapa 4: Confirmação PMT1/PMT2 e divisão Treino/Teste sem vazamento
# ==============================================================================


def confirmar_detectores(
    boca: pd.DataFrame,
    metodo: str,
    responsavel: str,
    observacao: str,
    force: bool,
) -> dict:
    if PMT_JSON.exists() and not force:
        print(f"[PULAR] Confirmação PMT1/PMT2: {PMT_JSON.name} já existe.")
        return json.loads(PMT_JSON.read_text(encoding="utf-8"))

    srs_cars = boca[boca["relative_path"].str.contains("SRS-CARS", na=False)].copy()
    pmt1_mods = sorted(srs_cars.loc[srs_cars["channel"] == "PMT1", "modality"].dropna().unique().tolist())
    pmt2_mods = sorted(srs_cars.loc[srs_cars["channel"] == "PMT2", "modality"].dropna().unique().tolist())

    inconsistencias = srs_cars[
        ((srs_cars["channel"] == "PMT1") & (srs_cars["modality"] != "CARS"))
        | ((srs_cars["channel"] == "PMT2") & (srs_cars["modality"] != "SRS"))
    ]
    if not inconsistencias.empty:
        raise RuntimeError(f"Encontradas {len(inconsistencias)} inconsistências PMT1/PMT2 em {IMAGES_CSV}!")

    payload = {
        "status": "confirmado",
        "convencao": {"PMT1_C0": "CARS", "PMT2_C1": "SRS"},
        "metodo_confirmacao": metodo,
        "responsavel": responsavel,
        "data_confirmacao_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "observacao": observacao,
        "auditoria_indice": {
            "total_arquivos_srs_cars": int(len(srs_cars)),
            "total_pares_exatos": int(srs_cars["pair_key"].nunique()),
            "modalidades_encontradas_PMT1": pmt1_mods,
            "modalidades_encontradas_PMT2": pmt2_mods,
            "inconsistencias_encontradas": 0,
        },
    }
    PMT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[OK] Confirmação dos detectores salva em: {PMT_JSON}")
    return payload


def gerar_split_sem_vazamento(
    boca: pd.DataFrame,
    test_ratio: float,
    seed: int,
    manter_11_01_no_treino: bool,
    force: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if SPLIT_IMAGES_CSV.exists() and SPLIT_CAMPOS_CSV.exists() and not force:
        print(f"[PULAR] Split sem vazamento: {SPLIT_IMAGES_CSV.name} e {SPLIT_CAMPOS_CSV.name} já existem.")
        return pd.read_csv(SPLIT_IMAGES_CSV), pd.read_csv(SPLIT_CAMPOS_CSV)

    df = boca.copy()
    df["condicao"] = df["relative_path"].apply(inferir_condicao)

    def montar_campo_id(row: pd.Series) -> str:
        if row["modality"] == "UNKNOWN" or pd.isna(row["sample_id"]):
            return "EXCLUDED_UNKNOWN"
        return f"{row['condicao']}__sample_{int(row['sample_id'])}"

    df["campo_group_id"] = df.apply(montar_campo_id, axis=1)

    anot = detectar_anotacoes()
    ids_anotados = anot["SHG"].union(anot["TPEF"])
    df["has_annotation"] = df["image_id"].astype(str).isin(ids_anotados)

    valid_df = df[df["campo_group_id"] != "EXCLUDED_UNKNOWN"].copy()
    campos = (
        valid_df.groupby(["condicao", "campo_group_id"], as_index=False)
        .agg(
            n_imagens=("image_id", "count"),
            n_anotadas=("has_annotation", "sum"),
            modalidades=("modality", lambda s: ",".join(sorted(set(s)))),
        )
        .sort_values(["condicao", "campo_group_id"])
        .reset_index(drop=True)
    )
    campos["tem_gabarito"] = campos["n_anotadas"] > 0

    rng = np.random.default_rng(seed)
    split_por_campo: dict[str, str] = {}

    for _, r in campos[campos["condicao"] == "11-01-2025"].iterrows():
        split_por_campo[r["campo_group_id"]] = "train" if manter_11_01_no_treino else "test"

    outros = campos[campos["condicao"] != "11-01-2025"]
    for (_, _), estrato in outros.groupby(["condicao", "tem_gabarito"]):
        perm = rng.permutation(estrato["campo_group_id"].tolist()).tolist()
        n_total = len(perm)
        n_test = max(1, int(round(n_total * test_ratio))) if n_total >= 2 else 0
        test_set = set(perm[:n_test])
        for cid in perm:
            split_por_campo[cid] = "test" if cid in test_set else "train"

    campos["split"] = campos["campo_group_id"].map(split_por_campo)
    df["split"] = df["campo_group_id"].map(split_por_campo).fillna("excluded")

    check = df[df["split"] != "excluded"].groupby("campo_group_id")["split"].nunique()
    if (check > 1).any():
        raise RuntimeError("Vazamento detectado: um mesmo campo_group_id caiu em treino e teste!")

    cols_img = [
        "image_id",
        "group",
        "subgroup",
        "condicao",
        "campo_group_id",
        "modality",
        "channel",
        "acquisition",
        "sample_id",
        "pair_key",
        "pair_quality",
        "has_annotation",
        "split",
        "relative_path",
    ]
    df_out = df[cols_img].sort_values(["split", "condicao", "campo_group_id", "modality", "acquisition"])
    df_out.to_csv(SPLIT_IMAGES_CSV, index=False)
    campos.to_csv(SPLIT_CAMPOS_CSV, index=False)

    print(f"[OK] Split por imagem travado em: {SPLIT_IMAGES_CSV}")
    print(f"[OK] Resumo por campo salvo em  : {SPLIT_CAMPOS_CSV}")
    return df_out, campos


def executar_finalizar(
    metodo_pmt: str = "configuracao_optica_equipamento",
    responsavel: str = "Gabriel",
    observacao_pmt: str = "PMT1/C0 confirmado como CARS e PMT2/C1 confirmado como SRS para aquisições simultâneas.",
    test_ratio: float = 0.35,
    seed: int = 42,
    force: bool = False,
) -> None:
    img_df = pd.read_csv(IMAGES_CSV)
    boca = img_df[img_df["group"] == "BOCA"].copy()

    confirmar_detectores(
        boca=boca,
        metodo=metodo_pmt,
        responsavel=responsavel,
        observacao=observacao_pmt,
        force=force,
    )
    gerar_split_sem_vazamento(
        boca=boca,
        test_ratio=test_ratio,
        seed=seed,
        manter_11_01_no_treino=True,
        force=force,
    )


def cmd_finalizar(args: argparse.Namespace) -> None:
    executar_finalizar(
        metodo_pmt=args.metodo_pmt,
        responsavel=args.responsavel,
        observacao_pmt=args.observacao_pmt,
        test_ratio=args.test_ratio,
        seed=args.seed,
        force=args.force,
    )
    cmd_status(args)


# ==============================================================================
# Etapa 5: Checklist final (status)
# ==============================================================================


def cmd_status(_: argparse.Namespace | None = None) -> None:
    print("\n=== Checklist de Conclusão — Atividade 1 (BOCA) ===")
    if MANIFESTO_CSV.exists():
        man = pd.read_csv(MANIFESTO_CSV)
        n_total = len(man)
        n_alinhados = int((man["alinhado"] == "sim").sum())
        print(
            f"[{'OK' if n_alinhados == n_total else 'PENDENTE'}] "
            f"Manifesto de pareamento: {n_alinhados}/{n_total} pares alinhados"
        )
    else:
        print("[PENDENTE] Manifesto de pareamento não encontrado")

    n_reg = len(list(REGISTERED_DIR.glob("candidate_*"))) if REGISTERED_DIR.exists() else 0
    print(f"[{'OK' if n_reg >= 26 else 'PARCIAL'}] Pares exportados em registered/BOCA: {n_reg}/26")

    anot = detectar_anotacoes()
    print(f"[OK] Gabarito anotado: {len(anot['SHG'])} máscaras SHG | {len(anot['TPEF'])} máscaras TPEF")
    print(f"[{'OK' if PMT_JSON.exists() else 'PENDENTE'}] Confirmação PMT1/PMT2: {PMT_JSON.name}")
    print(f"[{'OK' if SPLIT_IMAGES_CSV.exists() else 'PENDENTE'}] Split sem vazamento travado: {SPLIT_IMAGES_CSV.name}")


# ==============================================================================
# Orquestrador sequencial completo (executado por padrão)
# ==============================================================================


def cmd_pipeline(args: argparse.Namespace) -> None:
    print("Iniciando pipeline sequencial da Atividade 1 (etapas já concluídas serão puladas)...\n")

    # 1. Revisão visual de pares SHG/TPEF
    if not executar_revisar():
        return

    # 2. Registro espacial de deslocados e exportação dos 26 pares alinhados
    executar_registrar_automatico(mode="auto", forcar_exportacao=False)

    # 3. Anotação do padrão-ouro (pula se as máscaras alvo já existirem)
    concluido_anot, novas_mascaras = executar_anotar(max_pares=args.max_pares)
    if not concluido_anot:
        return

    # 4. Formalização PMT1/PMT2 e split treino/teste sem vazamento
    executar_finalizar(
        metodo_pmt=args.metodo_pmt,
        responsavel=args.responsavel,
        observacao_pmt=args.observacao_pmt,
        test_ratio=args.test_ratio,
        seed=args.seed,
        force=args.force or (novas_mascaras > 0),
    )

    # 5. Checklist final
    cmd_status()


# ==============================================================================
# CLI principal
# ==============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline consolidado e idempotente da Atividade 1 (BOCA).")
    parser.add_argument("--max-pares", type=int, default=12, help="Quantidade mínima de pares alvo no gabarito.")
    parser.add_argument("--metodo-pmt", default="configuracao_optica_equipamento")
    parser.add_argument("--responsavel", default="Gabriel")
    parser.add_argument(
        "--observacao-pmt",
        default="PMT1/C0 confirmado como CARS e PMT2/C1 confirmado como SRS para aquisições simultâneas.",
    )
    parser.add_argument("--test-ratio", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")

    sub = parser.add_subparsers(dest="comando")

    p_status = sub.add_parser("status", help="Exibe apenas o checklist da Atividade 1.")
    p_status.set_defaults(func=cmd_status)

    p_rev = sub.add_parser("revisar", help="Executa apenas a revisão visual dos pares.")
    p_rev.add_argument("--pair-id", default=None)
    p_rev.add_argument("--revisar-registrados", action="store_true")
    p_rev.set_defaults(func=cmd_revisar)

    p_reg = sub.add_parser("registrar", help="Executa apenas o registro/exportação dos pares.")
    p_reg.add_argument("--pair-id", default=None)
    p_reg.add_argument("--mode", choices=["auto", "landmarks", "identity"], default="auto")
    p_reg.add_argument("--forcar-exportacao", action="store_true")
    p_reg.set_defaults(func=cmd_registrar)

    p_ano = sub.add_parser("anotar", help="Abre o editor de máscaras do gabarito.")
    p_ano.add_argument("--max-pares", type=int, default=12)
    p_ano.add_argument("--pair-id", default=None)
    p_ano.set_defaults(func=cmd_anotar)

    p_fin = sub.add_parser("finalizar", help="Formaliza PMT1/PMT2 e gera o split treino/teste.")
    p_fin.add_argument("--metodo-pmt", default="configuracao_optica_equipamento")
    p_fin.add_argument("--responsavel", default="Gabriel")
    p_fin.add_argument(
        "--observacao-pmt",
        default="PMT1/C0 confirmado como CARS e PMT2/C1 confirmado como SRS para aquisições simultâneas.",
    )
    p_fin.add_argument("--test-ratio", type=float, default=0.35)
    p_fin.add_argument("--seed", type=int, default=42)
    p_fin.add_argument("--force", action="store_true")
    p_fin.set_defaults(func=cmd_finalizar)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        cmd_pipeline(args)
        return
    args.func(args)


if __name__ == "__main__":
    main()
