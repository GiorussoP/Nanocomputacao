from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import distance_transform_edt


def normalizar_contraste(img: np.ndarray) -> np.ndarray:
    img = np.squeeze(img).astype(np.float32)
    if img.ndim == 3:
        img = img[0]
    p_low, p_high = np.percentile(img, (1.0, 99.5))
    if p_high <= p_low:
        return np.zeros_like(img)
    return np.clip((img - p_low) / (p_high - p_low), 0, 1)


class EditorMascara:
    def __init__(self, img: np.ndarray, mask_inicial: np.ndarray, titulo: str):
        self.img = img
        # Trabalha com máscara binária (1 = estrutura/fibra, 0 = fundo)
        self.mask = (mask_inicial > 0).astype(np.uint8)
        self.raio = 8
        self.desenhando = False
        self.valor_pincel = 1  # 1 = pintar fibra, 0 = apagar
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
        self.overlay = self.ax_edit.imshow(
            self._criar_rgba(), interpolation="nearest"
        )
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
        rgba[self.mask > 0] = [0.0, 1.0, 0.3, 0.45]  # Verde semitransparente
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
        if event.button == 1:  # Botão esquerdo: pintar
            self.desenhando = True
            self.valor_pincel = 1
            self._aplicar_pincel(event.xdata, event.ydata)
        elif event.button == 3:  # Botão direito: apagar
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


def main() -> None:
    manifesto_path = Path("metadata/manifesto_pareamento_boca.csv")
    images_csv = Path("metadata/foundation_images.csv")
    seg_dir = Path("processed/foundation_microscopy/segmentations/BOCA")
    out_dir = Path("annotations/BOCA")

    manifesto = pd.read_csv(manifesto_path)
    df_imgs = pd.read_csv(images_csv)
    id_col = df_imgs.columns[0]
    path_col = "path" if "path" in df_imgs.columns else df_imgs.columns[1]
    mapa_caminhos = dict(zip(df_imgs[id_col].astype(str), df_imgs[path_col]))

    # Seleciona pares perfeitamente alinhados (ex: até 12 campos, balanceando condições)
    alinhados = manifesto[manifesto["alinhado"] == "sim"].copy()
    selecionados = alinhados.head(12)

    print(f"Iniciando anotação do padrão-ouro para {len(selecionados)} pares alinhados de BOCA.")
    print("Controles na janela:")
    print("  - Clique/Arraste ESQUERDO : Pintar estrutura (fibra)")
    print("  - Clique/Arraste DIREITO  : Apagar falso positivo (fundo)")
    print("  - Scroll ou teclas +/-    : Aumentar/diminuir pincel")
    print("  - Tecla C                 : Limpar máscara inteira (começar do zero)")
    print("  - Tecla ENTER             : Salvar máscara e ir para a próxima")
    print("  - Tecla Q                 : Sair\n")

    for idx, row in enumerate(selecionados.itertuples(), 1):
        for modalidade, img_id in [("SHG", str(row.shg_id)), ("TPEF", str(row.tpef_id))]:
            destino = out_dir / modalidade / f"{img_id}.tif"
            if destino.exists():
                print(f"[{idx}/{len(selecionados)}] {modalidade} ({img_id}) já anotado. Pulando...")
                continue

            raw_path = Path(mapa_caminhos[img_id])
            pred_path = seg_dir / modalidade / f"{img_id}.tif"

            img = normalizar_contraste(tifffile.imread(raw_path))
            if pred_path.exists():
                mask_init = np.squeeze(tifffile.imread(pred_path))
            else:
                mask_init = np.zeros_like(img, dtype=np.uint8)

            titulo = f"Campo {idx}/{len(selecionados)} ({row.condicao}) | {modalidade} | ID: {img_id}"
            editor = EditorMascara(img, mask_init, titulo)

            if editor.sair_tudo:
                print("Saindo do editor...")
                return

            if editor.salvo:
                destino.parent.mkdir(parents=True, exist_ok=True)
                tifffile.imwrite(destino, editor.mask.astype(np.uint8))
                print(f"  -> Salvo em: {destino}")

    print("\nGabarito concluído! Máscaras salvas em annotations/BOCA/{SHG,TPEF}/")


if __name__ == "__main__":
    main()