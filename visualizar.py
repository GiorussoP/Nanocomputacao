import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile


def normalizar_contraste(img: np.ndarray) -> np.ndarray:
    """Ajusta o brilho da imagem de 16 bits usando percentis (igual ao Auto do Fiji)."""
    img = np.squeeze(img).astype(np.float32)
    if img.ndim == 3:  # Caso tenha algum eixo extra (ex: tempo/canal)
        img = img[0]
    p_low, p_high = np.percentile(img, (1.0, 99.5))
    if p_high <= p_low:
        return np.zeros_like(img)
    return np.clip((img - p_low) / (p_high - p_low), 0, 1)


def mostrar_imagem_e_mascara(caminho_img: Path, caminho_mask: Path | None = None) -> None:
    img = normalizar_contraste(tifffile.imread(caminho_img))

    if caminho_mask is None:
        plt.figure(figsize=(8, 8))
        plt.imshow(img, cmap="gray")
        plt.title(caminho_img.name, fontsize=10)
        plt.axis("off")
    else:
        mask = np.squeeze(tifffile.imread(caminho_mask))
        # Cria uma camada colorida transparente onde mask > 0
        mask_vis = np.ma.masked_where(mask == 0, mask)

        fig, axes = plt.subplots(1, 3, figsize=(16, 6))
        axes[0].imshow(img, cmap="gray")
        axes[0].set_title("Imagem Original (Contrastada)")

        axes[1].imshow(mask_vis, cmap="nipy_spectral", interpolation="nearest")
        axes[1].set_title(f"Máscara ({len(np.unique(mask)) - 1} objetos)")

        axes[2].imshow(img, cmap="gray")
        axes[2].imshow(mask_vis, cmap="nipy_spectral", alpha=0.45, interpolation="nearest")
        axes[2].set_title("Sobreposição (Overlay)")

        for ax in axes:
            ax.axis("off")

    plt.tight_layout()
    plt.show()


def comparar_alinhamento(caminho_a: Path, caminho_b: Path) -> None:
    """Sobrepõe Canal A (Verde) e Canal B (Magenta) para checar co-registro."""
    img_a = normalizar_contraste(tifffile.imread(caminho_a))
    img_b = normalizar_contraste(tifffile.imread(caminho_b))

    if img_a.shape != img_b.shape:
        raise ValueError(f"Tamanhos diferentes: {img_a.shape} vs {img_b.shape}")

    # RGB: R=Canal B, G=Canal A, B=Canal B (onde coincidem fica branco)
    rgb = np.stack([img_b, img_a, img_b], axis=-1)

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    axes[0].imshow(img_a, cmap="Greens")
    axes[0].set_title(f"Canal A (Verde): {caminho_a.name[:25]}...")

    axes[1].imshow(img_b, cmap="RdPu")
    axes[1].set_title(f"Canal B (Magenta): {caminho_b.name[:25]}...")

    axes[2].imshow(rgb)
    axes[2].set_title("Sobreposição (Branco = Alinhado)")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualizador rápido de microscopia OME-TIFF e máscaras.")
    parser.add_argument("imagem", type=Path, help="Caminho para a imagem .tif / .ome.tif")
    parser.add_argument("--mask", type=Path, default=None, help="Caminho opcional para a máscara segmentada")
    parser.add_argument("--compare", type=Path, default=None, help="Segunda imagem para checar alinhamento")
    args = parser.parse_args()

    if args.compare:
        comparar_alinhamento(args.imagem, args.compare)
    else:
        mostrar_imagem_e_mascara(args.imagem, args.mask)