"""Atividade 4 — padrões visuais de CARS e SRS em BOCA e Walleriana.

Em BOCA, os dois canais de cada par simultâneo são analisados juntos. Em
Walleriana, CARS e SRS são analisados separadamente. Curvelet, Otsu e
componentes conexos delimitam objetos para medir forma e intensidade original.
Cada campo fornece 16 recortes (4 × 4). O k-means agrupa os mesmos recortes
por descritores clássicos ou por embeddings DINOv2, em ajustes independentes.

Como executar a partir de Nanocomputacao/dados_raw
    conda activate foundation
    python atividade_4.py --clusters 4                 # k fixo (padrão 4)
    python atividade_4.py --k-classico 3 --k-dino 5   # k distinto por método
    python atividade_4.py --k-automatico --k-min 2 --k-max 8
    python atividade_4.py --teste-manual              # 10 primeiros pares BOCA
    python atividade_4.py --teste-manual-full         # todos os pares BOCA válidos
    python atividade_4.py --somente-visualizacoes --output processed/atividade_4_k4

Saídas da análise regular em processed/ (ou na pasta indicada por --output)
    campos.csv e objetos.csv — medidas por imagem e por objeto segmentado.
    pares.csv — Dice e IoU entre máscaras e correlação de intensidade CARS/SRS.
    patches.csv — posição, medidas e os dois clusters de cada recorte.
    perfis.csv — uma linha por grupo, método e cluster, com medianas das
        medidas dos recortes, contagem de recortes e campos e exemplo central.
    comparacao_clusters.csv — concordância entre métodos (ARI).
    walleriana.csv — medidas CARS por pasta e comparação com o controle.
    exclusoes.csv — campos excluídos e motivos.
    selecao_k.csv — no modo automático, silhueta, estabilidade (ARI médio
        em dez reamostragens por aquisição) e k escolhido pela média das duas.
    mascaras/ e sobreposicoes/ — objetos rotulados e canais BOCA sobrepostos.
    PNGs de exemplos e visualizacoes/ — recortes representativos, gráficos
        de pontos, mapa de calor BOCA e mapas espaciais individuais e globais.
    --somente-visualizacoes relê campos.csv e patches.csv de uma análise
        regular e usa os TIFFs originais para redesenhar os mapas e gráficos.

Testes manuais de BOCA, com os mesmos pares em todos os ajustes de k=2 a 8
    --teste-manual gera uma pasta por par em atividade_4_teste_manual/.
    --teste-manual-full gera atividade_4_teste_manual_full/kN/ com
        BOCA_global_kN.png e perfis.csv. No atlas, o clássico fica à esquerda
        e o DINOv2 à direita. Cruze método e rótulo com o CSV da mesma pasta.
        As frações indicam cobertura segmentada e os componentes contam objetos
        que tocam o recorte. Rótulos podem mudar de significado entre ajustes.
"""
import argparse
import hashlib
from functools import lru_cache
from pathlib import Path

from matplotlib import colormaps
from matplotlib.figure import Figure
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd
import tifffile
import torch
from PIL import Image, ImageDraw
from curvelets.numpy import UDCT
from scipy import ndimage as ndi
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, roc_auc_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from skimage.filters import threshold_otsu
from skimage.measure import regionprops
from threadpoolctl import threadpool_limits

from foundation_microscopy.imaging import normalize
from foundation_microscopy.inventory import read_csv


ROOT = Path(__file__).resolve().parent
GRID = 4


def image_relative_path(filename):
    path = Path(filename)
    prefix = Path('raw/IMAGENS')
    return path.relative_to(prefix) if path.is_relative_to(prefix) else path


def local_image(filename):
    return ROOT / 'raw/IMAGENS' / image_relative_path(filename)


def acquisitions(records):
    pairs = {}
    fields = []
    excluded = []
    for r in records:
        if r.modality not in ('CARS', 'SRS'):
            continue
        if r.group == 'BOCA' and r.pair_quality == 'exact':
            pairs.setdefault(r.pair_key, []).append(r)
        if 'WALLERIANA' in r.group.upper():
            parts = image_relative_path(r.relative_path).parts
            if len(parts) >= 3 and parts[1] == 'NO' and parts[2] in ('control', '7', '14'):
                fields.append(('NO_' + r.modality, parts[2], r.image_id, [r]))
    for key, members in sorted(pairs.items()):
        members.sort(key=lambda r: r.modality)
        if ([r.modality for r in members] != ['CARS', 'SRS'] or
                any('SRS-CARS' not in Path(r.relative_path).name.upper() for r in members)):
            excluded.append(dict(campo=key, motivo='canais ou nomes inconsistentes'))
        else:
            fields.append(('BOCA', 'BOCA', key, members))
    return fields, excluded


def load_model(device):
    cache = Path(torch.hub.get_dir())
    repo = cache / 'facebookresearch_dinov2_main'
    weights = cache / 'checkpoints/dinov2_vits14_pretrain.pth'
    if repo.exists() and weights.is_file():
        model = torch.hub.load(str(repo), 'dinov2_vits14', source='local', pretrained=False)
        model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True))
    else:
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    return model.eval().to(device)


@torch.inference_mode()
def encode(model, crops, device):
    batch = np.stack([np.asarray(c.resize((224, 224), Image.Resampling.BICUBIC)) for c in crops])
    tensor = torch.from_numpy(batch).to(device)[:, None].float() / 255
    mean = tensor.new_tensor([.485, .456, .406])[None, :, None, None]
    std = tensor.new_tensor([.229, .224, .225])[None, :, None, None]
    features = model((tensor - mean) / std)
    return torch.nn.functional.normalize(features).cpu().numpy()


@lru_cache(maxsize=1)
def transform(shape):
    return UDCT(shape, num_scales=3, wedges_per_direction=3,
                high_frequency_mode='curvelet')


def curvelet_filter(gray):
    # Reflexão nas bordas e dimensões divisíveis pela decimação da UDCT.
    margin = 16
    pads = [(margin, margin + (-(n + 2 * margin)) % 32) for n in gray.shape]
    extended = np.pad(gray, pads, mode='reflect')
    udct = transform(extended.shape)
    coeffs = udct.forward(extended)
    # Preserva a aproximação. Em cada cunha direcional, contrai o módulo
    # pelo limiar de duas medianas da banda e preserva a fase complexa.
    for scale in coeffs[1:]:
        for direction in scale:
            for band in direction:
                magnitude = np.abs(band)
                threshold = 2 * np.median(magnitude)
                band *= np.maximum(0, 1 - threshold / np.maximum(magnitude, 1e-20))
    filtered = udct.backward(coeffs)
    return filtered[margin:margin+gray.shape[0], margin:margin+gray.shape[1]]


def components(smoothed, min_area):
    if smoothed.max() == smoothed.min():
        return np.zeros(smoothed.shape, dtype=np.int32)
    labels, _ = ndi.label(smoothed > threshold_otsu(smoothed), np.ones((3, 3)))
    counts = np.bincount(labels.ravel())
    return ndi.label((labels > 0) & (counts[labels] >= min_area), np.ones((3, 3)))[0]


def fractions(a, b):
    intersection = np.count_nonzero(a & b)
    total = a.sum() + b.sum()
    return dict(dice=2 * intersection / total if total else np.nan,
                iou=intersection / (total - intersection) if total else np.nan)


def analyze(field, model, device, min_area, tables, features, previews, output):
    group, condition, key, members = field
    base = dict(grupo=group, pasta=condition, campo=key)
    images = {}
    gray = {}
    labels = {}
    filtered = {}
    for r in members:
        with tifffile.TiffFile(local_image(r.relative_path)) as f:
            # Uma série OME pode referenciar os dois arquivos PMT. A página
            # física pertence ao arquivo do detector indicado no inventário.
            image = f.pages[0].asarray().squeeze()
        if image.ndim != 2 or (images and image.shape != next(iter(images.values())).shape):
            tables['exclusoes'].append(dict(campo=key, motivo='geometria incompatível'))
            return
        images[r.modality] = image
        gray[r.modality] = normalize(image)
        filtered[r.modality] = curvelet_filter(gray[r.modality])
        labels[r.modality] = components(filtered[r.modality], min_area)
    h, w = image.shape
    ys, xs = (np.linspace(0, n, GRID + 1, dtype=int) for n in (h, w))
    regions = [(slice(y, yy), slice(x, xx)) for y, yy in zip(ys, ys[1:])
               for x, xx in zip(xs, xs[1:])]
    patches = [dict(**base, patch=i, y=s[0].start, x=s[1].start,
                    altura=s[0].stop-s[0].start, largura=s[1].stop-s[1].start)
               for i, s in enumerate(regions)]
    encoded, thumbs = [], []
    for r in members:
        ch = r.modality
        lab = labels[ch]
        if output is not None:
            tifffile.imwrite(output / 'mascaras' / f'{r.image_id}.tif',
                             lab.astype('uint32'), compression='zlib')
        props = regionprops(lab)
        lengths, angles = [], []
        for p in props:
            y, x = p.centroid
            patch = (np.searchsorted(ys[1:], y, side='right') * GRID
                     + np.searchsorted(xs[1:], x, side='right'))
            row = dict(**base, origem=ch, imagem=r.image_id, objeto=p.label, patch=patch,
                       y=y, x=x, area_px=p.area, eixo_maior_px=p.axis_major_length,
                       excentricidade=p.eccentricity, orientacao_rad=p.orientation)
            lengths.append(row['eixo_maior_px'])
            if p.eccentricity >= .75:
                angles.append(p.orientation)
            for channel, raw in images.items():
                inside = raw[p.slice][p.image]
                row['mediana_' + channel] = np.median(inside)
                row['media_' + channel] = inside.mean()
                row['cobertura_' + channel] = (labels[channel][p.slice][p.image] > 0).mean()
            tables['objetos'].append(row)
        summary = dict(**base, modalidade=ch, imagem=r.image_id, arquivo=r.relative_path,
                       hash_plano=hashlib.sha256(images[ch].tobytes()).hexdigest(),
                       altura=h, largura=w, objetos=len(props), objetos_por_Mpx=len(props)*1e6/(w*h),
                       fracao_positiva=(lab > 0).mean(),
                       eixo_mediano_px=np.median(lengths) if lengths else np.nan,
                       dispersao_axial=1-abs(np.exp(2j*np.array(angles)).mean()) if angles else np.nan)
        tables['campos'].append(summary)
        crops = []
        for row, s in zip(patches, regions):
            tile = gray[ch][s]
            row['media_norm_' + ch] = tile.mean()
            row['desvio_norm_' + ch] = tile.std()
            row['fracao_' + ch] = (lab[s] > 0).mean()
            row['residuo_curvelet_' + ch] = np.sqrt(np.mean((tile - filtered[ch][s])**2))
            row['componentes_' + ch] = np.unique(lab[s][lab[s] > 0]).size
            crops.append(Image.fromarray(np.round(255 * tile).astype('uint8')))
        encoded.append(encode(model, crops, device))
        thumbs.append([c.resize((80, 80)) for c in crops])
    if len(members) == 2 and output is not None:
        cars, srs = (images[c].ravel() for c in ('CARS', 'SRS'))
        corr = np.corrcoef(cars, srs)[0, 1] if cars.std() and srs.std() else np.nan
        tables['pares'].append(dict(**base, correlacao_intensidades=corr,
                                    **fractions(labels['CARS'] > 0, labels['SRS'] > 0)))
        # SRS em magenta, CARS em verde, na geometria nativa da aquisição.
        rgb = np.stack([gray['SRS'], gray['CARS'], gray['SRS']], axis=-1)
        overlay = Image.fromarray(np.round(255 * rgb).astype('uint8'))
        overlay.thumbnail((512, 512))
        overlay.save(output / 'sobreposicoes' / (key.replace(':', '_') + '.png'))
    tables['patches'].extend(patches)
    features.extend(np.concatenate(encoded, axis=1))
    previews.extend(zip(*thumbs))


def stability_splits(fields, repeats=10):
    """Dois treinos parcialmente sobrepostos e um teste comum fora dos treinos."""
    rng = np.random.default_rng(42)
    ids = np.unique(fields)
    if len(ids) < 4:
        raise ValueError('Estabilidade por aquisição requer pelo menos quatro campos.')
    ntest = max(1, int(np.ceil(.2 * len(ids))))
    ntrain = max(2, int(.8 * (len(ids) - ntest)))
    for _ in range(repeats):
        shuffled = rng.permutation(ids)
        test = np.isin(fields, shuffled[:ntest])
        pool = shuffled[ntest:]
        # Ao menos um campo diferente em cada treino.
        a = np.isin(fields, pool[:ntrain])
        b = np.isin(fields, pool[-ntrain:])
        yield a, b, test


def select_model(raw, method, group, fields, candidates):
    # No clássico, cada treino estima a padronização apenas com seus campos.
    matrix = StandardScaler().fit_transform(raw) if method == 'classico' else raw
    splits = []
    for first, second, test in stability_splits(fields):
        trials = []
        for mask in (first, second):
            train, heldout = raw[mask], raw[test]
            if method == 'classico':
                scaler = StandardScaler().fit(train)
                train, heldout = scaler.transform(train), scaler.transform(heldout)
            trials.append((train, heldout))
        splits.append(trials)
    records = []
    best = None
    for k in candidates:
        if k >= len(matrix) or any(len(train) < k for trial in splits for train, _ in trial):
            continue
        fit = KMeans(n_clusters=k, random_state=42, n_init=10).fit(matrix)
        if np.unique(fit.labels_).size != k:
            continue
        sil = silhouette_score(matrix, fit.labels_)
        agreement = []
        for i, trial in enumerate(splits):
            predictions = []
            for j, (train, heldout) in enumerate(trial):
                model = KMeans(n_clusters=k, random_state=100+2*i+j, n_init=10).fit(train)
                predictions.append(model.predict(heldout))
            agreement.append(adjusted_rand_score(*predictions))
        stability = np.mean(agreement)
        score = (sil + stability) / 2
        records.append(dict(grupo=group, metodo=method, k=k, silhueta=sil,
                            estabilidade=stability, pontuacao=score, repeticoes=len(agreement)))
        # Candidatos crescentes: empate exato favorece o menor k.
        if best is None or score > best[0]:
            best = score, fit
    if best is None:
        raise ValueError(f'Nenhum k avaliável em {group}/{method} para o intervalo solicitado.')
    winner = best[1]
    for record in records:
        record.update(melhor_k=winner.n_clusters, selecionado=record['k'] == winner.n_clusters)
    return matrix, winner, records


def classic_columns(rows):
    prefixes = ('media_norm_', 'desvio_norm_', 'fracao_', 'residuo_curvelet_', 'componentes_')
    return [c for c in rows if c.startswith(prefixes) and rows[c].notna().all()]


def fixed_kmeans(matrix, k):
    return KMeans(n_clusters=min(k, len(matrix)), random_state=42, n_init=10).fit(matrix)


def cluster_profiles(rows, matrix, fit, group, method, cols):
    """Resume cada cluster e identifica o recorte mais próximo do centro ajustado."""
    for label in np.unique(fit.labels_):
        pos = np.flatnonzero(fit.labels_ == label)
        subset = rows.iloc[pos]
        distances = np.linalg.norm(matrix[pos] - fit.cluster_centers_[label], axis=1)
        example = subset.iloc[np.argmin(distances)]
        profile = dict(grupo=group, metodo=method, cluster=label, patches=len(subset),
                       campos=subset.campo.nunique(), campo_exemplo=example.campo,
                       patch_exemplo=example.patch, **subset[cols].median().to_dict())
        yield profile, example.name


def cluster(tables, features, previews, output, k, *, k_dino=None, candidates=None):
    df = pd.DataFrame(tables['patches'])
    profiles, comparisons, selection = [], [], []
    for group, rows in df.groupby('grupo'):
        cols = classic_columns(rows)
        classic = rows[cols].to_numpy()
        deep = np.stack([features[i] for i in rows.index])
        assignments = []
        for method, matrix in [('classico', classic), ('dino', deep)]:
            if candidates is not None:
                matrix, fit, scores = select_model(matrix, method, group, rows.campo.to_numpy(), candidates)
                selection.extend(scores)
            else:
                n = k_dino if method == 'dino' and k_dino is not None else k
                if method == 'classico':
                    matrix = StandardScaler().fit_transform(matrix)
                fit = fixed_kmeans(matrix, n)
            assignments.append(fit.labels_)
            df.loc[rows.index, 'cluster_' + method] = fit.labels_
            for profile, idx in cluster_profiles(rows, matrix, fit, group, method, cols):
                profiles.append(profile)
                label = profile['cluster']
                tiles = previews[idx]
                sheet = Image.new('RGB', (80*len(tiles), 100))
                draw = ImageDraw.Draw(sheet)
                channels = ['CARS','SRS'] if group == 'BOCA' else [group.split('_')[-1]]
                for j, tile in enumerate(tiles):
                    sheet.paste(tile, (80*j, 20))
                    draw.text((80*j, 2), channels[j], fill='white')
                sheet.save(output / f'{group}_{method}_{label}.png')
        comparisons.append(dict(grupo=group, ARI=adjusted_rand_score(*assignments)))
    objects = pd.DataFrame(tables['objetos'])
    keys = ['grupo','campo','patch']
    if not objects.empty:
        objects = objects.merge(df[keys + ['cluster_dino','cluster_classico']], on=keys,
                                how='left', validate='many_to_one')
    results = dict(patches=df, objetos=objects, perfis=pd.DataFrame(profiles),
                   comparacao_clusters=pd.DataFrame(comparisons))
    if selection:
        results['selecao_k'] = pd.DataFrame(selection)
    return results


def unique_acquisitions(subset):
    across = subset.groupby('hash_plano').pasta.transform('nunique')
    return subset.loc[across == 1].drop_duplicates('hash_plano')


def walleriana(fields):
    subset = unique_acquisitions(fields.loc[fields.grupo == 'NO_CARS'])
    metrics = ['objetos_por_Mpx', 'eixo_mediano_px', 'dispersao_axial']
    summary = subset.groupby('pasta')[metrics].median()
    summary['aquisicoes'] = subset.groupby('pasta').size()
    if 'control' in summary.index:
        for m in metrics:
            summary['delta_control_' + m] = summary[m] - summary.loc['control', m]
            # AUC descritiva: hipótese prévia de maior densidade e dispersão,
            # e menor comprimento. Cada aquisição contribui com uma observação.
            for condition in summary.index.difference(['control']):
                comparison = subset[subset.pasta.isin(['control', condition])].dropna(subset=[m])
                target = comparison.pasta == condition
                if target.nunique() == 2:
                    score = comparison[m] * (-1 if m == 'eixo_mediano_px' else 1)
                    summary.loc[condition, 'auc_' + m] = roc_auc_score(target, score)
    return summary.reset_index()


def compare_examples(output, profiles, patches):
    """Mostra os dois rótulos independentes atribuídos a cada exemplo BOCA."""
    boca = patches[patches.grupo == 'BOCA'].set_index(['campo', 'patch'])
    examples = profiles[profiles.grupo == 'BOCA']
    if examples.empty:
        return
    sheet = Image.new('RGB', (410, 110 * len(examples)), 'white')
    draw = ImageDraw.Draw(sheet)
    for i, example in enumerate(examples.itertuples()):
        row = boca.loc[(example.campo_exemplo, example.patch_exemplo)]
        y = 110 * i
        with Image.open(output / f'BOCA_{example.metodo}_{example.cluster}.png') as pair:
            sheet.paste(pair, (0, y))
        text = (f'Exemplo do {example.metodo}, grupo {example.cluster}\n'
                f'Classico {int(row.cluster_classico)} | DINOv2 {int(row.cluster_dino)}\n'
                f'{example.campo_exemplo}\nPatch {example.patch_exemplo}')
        draw.multiline_text((170, y + 10), text, fill='black', spacing=4)
    sheet.save(output / 'BOCA_comparacao.png')


def scatter_plots(fields, folder):
    boca = fields[fields.grupo == 'BOCA']
    coverage = 100 * boca.pivot(index='campo', columns='modalidade', values='fracao_positiva')
    fig = Figure(figsize=(6, 5.5), layout='constrained')
    ax = fig.subplots()
    ax.plot([0, 100], [0, 100], '--', color='.55', label='Mesma cobertura')
    ax.scatter(coverage.CARS, coverage.SRS, color='#007889', alpha=.75, edgecolors='white')
    ax.set(xlim=(-2, 102), ylim=(-2, 102), aspect='equal', xlabel='Área positiva em CARS (%)',
           ylabel='Área positiva em SRS (%)', title=f'BOCA · Cobertura em {len(coverage)} pares')
    ax.legend(frameon=False)
    fig.savefig(folder / 'BOCA_pontos.png', dpi=160)

    fig = Figure(figsize=(11, 4.2), layout='constrained')
    axes = fig.subplots(1, 3)
    fig.suptitle('Walleriana · Um ponto por aquisição, traço na mediana')
    metrics = [('objetos_por_Mpx', 'Objetos por megapixel'), ('eixo_mediano_px', 'Eixo maior mediano (px)'),
               ('dispersao_axial', 'Dispersão axial')]
    conditions = ['control', '7', '14']
    rng = np.random.default_rng(42)
    for channel, offset, color, marker in [('CARS', -.10, '#126984', 'o'), ('SRS', .10, '#ab4a20', '^')]:
        subset = unique_acquisitions(fields[fields.grupo == 'NO_' + channel])
        for i, condition in enumerate(conditions):
            data = subset[subset.pasta.astype(str) == condition]
            if data.empty:
                continue
            x = i + offset + rng.uniform(-.07, .07, len(data))
            for ax, (metric, title) in zip(axes, metrics):
                ax.scatter(x, data[metric], color=color, marker=marker, alpha=.68, s=23)
                ax.hlines(data[metric].median(), i+offset-.09, i+offset+.09, color=color, linewidth=2.8)
                ax.set(ylabel=title, xlabel='Pasta experimental', xlim=(-.45, 2.45))
                ax.set_xticks(range(3), conditions)
        axes[0].scatter([], [], color=color, marker=marker, label=f'{channel} (n={len(subset)})')
    axes[0].legend(frameon=False)
    fig.savefig(folder / 'Walleriana_pontos.png', dpi=160)


def cluster_heatmap(boca, folder):
    counts = pd.crosstab(boca.cluster_classico, boca.cluster_dino)
    percent = counts.div(counts.sum(axis=1), axis=0) * 100
    fig = Figure(figsize=(max(6, .9*counts.shape[1]), max(5, .7*len(counts))), layout='constrained')
    ax = fig.subplots()
    heat = ax.imshow(percent, cmap='Blues', vmin=0, vmax=100)
    for i, j in np.ndindex(counts.shape):
        value = percent.iat[i, j]
        ax.text(j, i, f'{counts.iat[i,j]}\n{value:.1f}%', ha='center', va='center',
                color='white' if value > 55 else '#152333')
    ax.set_xticks(range(counts.shape[1]), counts.columns)
    ax.set_yticks(range(len(counts)), counts.index)
    ax.set(xlabel='Cluster DINOv2', ylabel='Cluster clássico',
           title=f'BOCA · Atribuições dos mesmos {len(boca)} recortes')
    fig.colorbar(heat, ax=ax, label='Percentual dentro de cada cluster clássico')
    fig.savefig(folder / 'BOCA_calor.png', dpi=160)


MAP_METHODS = [('classico', 'Clássico', 'viridis'), ('dino', 'DINOv2', 'plasma')]


def cluster_palettes(boca):
    palettes = {}
    for method, _, cmap in MAP_METHODS:
        labels = sorted(boca['cluster_' + method].unique())
        colors = colormaps[cmap](np.linspace(.12, .86, len(labels)))
        palettes[method] = dict(zip(labels, colors))
    return palettes


def draw_field(rows, paths, palettes, title, destination):
    channels = list(paths)
    panels = len(channels) + len(MAP_METHODS)
    fig = Figure(figsize=(3.125 * panels, 4.1), layout='constrained')
    axes = fig.subplots(1, panels)
    fig.suptitle(title)
    images = []
    for ax, channel in zip(axes, channels):
        source = paths[channel]
        with tifffile.TiffFile(source) as tif:
            images.append(normalize(tif.pages[0].asarray().squeeze()))
        ax.imshow(images[-1], cmap='gray', vmin=0, vmax=1)
        ax.set_title(channel)
    background = np.mean(images, axis=0)
    for ax, (method, title, _) in zip(axes[len(channels):], MAP_METHODS):
        ax.imshow(background, cmap='gray', vmin=0, vmax=1)
        ax.set_title(title)
        for row in rows.itertuples():
            label = getattr(row, 'cluster_' + method)
            color = palettes[method][label]
            ax.add_patch(Rectangle((row.x-.5, row.y-.5), row.largura, row.altura,
                                   facecolor=(*color[:3], .42), edgecolor='white', linewidth=.65))
            ax.text(row.x+row.largura/2, row.y+row.altura/2, str(label),
                    ha='center', va='center', color='white',
                    bbox=dict(facecolor='black', edgecolor='none', alpha=.65, pad=1))
    for ax in axes:
        ax.set_axis_off()
    fig.savefig(destination, dpi=145)


def global_map(patches, palettes, title, destination, *, show_condition=False):
    legend = [Patch(color=color, label=f'{title} {label}')
              for method, title, _ in MAP_METHODS for label, color in palettes[method].items()]
    groups = patches.groupby('campo', sort=True)
    ncols = min(5, len(groups))
    nrows = (len(groups) + ncols - 1) // ncols
    atlas = Figure(figsize=(ncols * 2.2, nrows * 1.45 + 1.4))
    atlas.suptitle(f'{title}\nEsquerda: clássico    |    Direita: DINOv2', y=.994)
    atlas.subplots_adjust(left=.025, right=.975, top=1-.84/atlas.get_figheight(),
                         bottom=.6/atlas.get_figheight(), hspace=.35)
    for index, (key, rows) in enumerate(groups, 1):
        short = key.split(':')[-1][:8]
        condition = f' · {rows.pasta.iloc[0]}' if show_condition else ''
        grids = []
        for method, _, _ in MAP_METHODS:
            grid = rows.pivot(index='y', columns='x', values='cluster_' + method)
            grids.append(np.array([[palettes[method][v] for v in line] for line in grid.to_numpy()]))
        ax = atlas.add_subplot(nrows, ncols, index)
        # Uma coluna branca separa os dois mapas do mesmo campo.
        gap = np.ones((grids[0].shape[0], 1, 4))
        ax.imshow(np.concatenate([grids[0], gap, grids[1]], axis=1), interpolation='nearest')
        ax.set_title(f'{index:02d}{condition} · {short}', fontsize=9)
        ax.set_axis_off()
    atlas.legend(handles=legend, loc='lower center', ncol=min(8,len(legend)), frameon=False)
    atlas.savefig(destination, dpi=160, bbox_inches='tight')


def spatial_maps(fields, patches, folder):
    for group, subset in patches.groupby('grupo'):
        is_boca = group == 'BOCA'
        group_fields = fields.loc[fields.grupo == group]
        channels = sorted(group_fields.modalidade.unique())
        sources = group_fields.set_index(['campo', 'modalidade'])
        name = 'BOCA' if is_boca else f'Walleriana_{channels[0]}'
        subdir = folder / ('pares' if is_boca else name)
        subdir.mkdir(exist_ok=True)
        palettes = cluster_palettes(subset)
        for index, (key, rows) in enumerate(subset.groupby('campo'), 1):
            paths = {}
            for channel in channels:
                filename = sources.loc[(key, channel), 'arquivo']
                paths[channel] = local_image(filename)
            title = (f'BOCA · Par {index:02d}' if is_boca else
                     f'Walleriana · {channels[0]} · Pasta {rows.pasta.iloc[0]} · Imagem {index:02d}')
            draw_field(rows, paths, palettes, f'{title} · {key}',
                       subdir / f'{index:02d}_{key.split(":")[-1]}.png')
        unit = 'par' if is_boca else 'imagem'
        global_map(subset, palettes, f'{name.replace("_", " · ")} · Mapas por {unit}',
                   folder / f'{name}_global.png', show_condition=not is_boca)


def visualizations(output, fields, patches):
    folder = output / 'visualizacoes'
    folder.mkdir(exist_ok=True)
    patches = patches.astype({'cluster_classico': int, 'cluster_dino': int})
    boca = patches.loc[patches.grupo == 'BOCA']
    scatter_plots(fields, folder)
    cluster_heatmap(boca, folder)
    spatial_maps(fields, patches, folder)
    print(f'Visualizações em {folder}')


def manual_pair_test(fields, model, device, min_area, output, *, full=False):
    """Compara sete valores de k sobre a mesma seleção de pares BOCA."""
    tables = {name: [] for name in ('patches', 'campos', 'objetos', 'pares', 'exclusoes')}
    features, previews, pairs = [], [], []
    for field in fields:
        group, _, key, members = field
        if group != 'BOCA':
            continue
        before = len(features)
        analyze(field, model, device, min_area, tables, features, previews, None)
        if len(features) == before:
            continue
        paths = {r.modality: local_image(r.relative_path) for r in members}
        pairs.append((key, paths))
        if full and len(pairs) % 10 == 0:
            print(f'{len(pairs)} pares BOCA preparados', flush=True)
        if not full and len(pairs) == 10:
            break
    if not pairs:
        raise ValueError('Nenhum par BOCA válido para o teste manual.')
    if not full and len(pairs) != 10:
        raise ValueError(f'O teste manual requer dez pares BOCA válidos; encontrados: {len(pairs)}.')

    patches = pd.DataFrame(tables['patches'])
    cols = classic_columns(patches)
    classic = StandardScaler().fit_transform(patches[cols])
    deep = np.stack(features)
    for k in range(2, 9):
        profiles = []
        with threadpool_limits(limits=2):
            for method, matrix in [('classico', classic), ('dino', deep)]:
                fit = fixed_kmeans(matrix, k)
                patches['cluster_' + method] = fit.labels_
                if full:
                    profiles.extend(profile for profile, _ in
                                    cluster_profiles(patches, matrix, fit, 'BOCA', method, cols))
        palettes = cluster_palettes(patches)
        if full:
            folder = output / f'k{k}'
            folder.mkdir(parents=True, exist_ok=True)
            global_map(patches, palettes, f'BOCA · {len(pairs)} pares · k={k}',
                       folder / f'BOCA_global_k{k}.png')
            pd.DataFrame(profiles).to_csv(folder / 'perfis.csv', index=False)
            print(f'k={k}: atlas dos {len(pairs)} pares e perfis em {folder}', flush=True)
            continue
        for index, (key, paths) in enumerate(pairs, 1):
            folder = output / f'{index:02d}_{key.split(":")[-1]}'
            folder.mkdir(parents=True, exist_ok=True)
            rows = patches.loc[patches.campo == key]
            title = f'BOCA · Par {index:02d} · {key} · k={k}\n{paths["CARS"].parent.name}'
            draw_field(rows, paths, palettes, title, folder / f'k{k}.png')
        print(f'k={k}: mapas dos dez pares gravados em {output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, help='Pasta de saída. Com k fixo comum, o padrão é processed/atividade_4_kN.')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--clusters', type=int, default=4, help='K fixo comum (padrão: 4).')
    mode.add_argument('--teste-manual', action='store_true',
                      help='Gera mapas dos dez primeiros pares válidos de BOCA para k de 2 a 8.')
    mode.add_argument('--teste-manual-full', action='store_true',
                      help='Gera um atlas global de BOCA e um perfis.csv em cada pasta de k2 a k8.')
    parser.add_argument('--k-classico', type=int, help='Substitui --clusters apenas no clássico.')
    parser.add_argument('--k-dino', type=int, help='Substitui --clusters apenas no DINOv2.')
    mode.add_argument('--k-automatico', action='store_true',
                      help='Maximiza a média de silhueta e estabilidade por método e conjunto.')
    parser.add_argument('--k-min', type=int, default=2, help='Primeiro k automático (padrão: 2).')
    parser.add_argument('--k-max', type=int, default=8, help='Último k automático (padrão: 8).')
    parser.add_argument('--min-area', type=int, default=32)
    parser.add_argument('--somente-visualizacoes', action='store_true',
                        help='Gera as figuras a partir de campos.csv e patches.csv em --output.')
    args = parser.parse_args()
    manual = args.teste_manual or args.teste_manual_full
    if manual and (args.somente_visualizacoes or args.k_classico is not None
                              or args.k_dino is not None or (args.k_min, args.k_max) != (2, 8)):
        parser.error('--teste-manual usa k de 2 a 8 para os dois métodos e gera os próprios mapas.')
    if args.k_automatico and (args.k_classico is not None or args.k_dino is not None):
        parser.error('--k-automatico é incompatível com --k-classico ou --k-dino')
    kc = args.clusters if args.k_classico is None else args.k_classico
    kd = args.clusters if args.k_dino is None else args.k_dino
    if min(kc, kd, args.clusters, args.min_area) < 1 or not 2 <= args.k_min <= args.k_max:
        parser.error('Use k e min-area positivos e 2 <= k-min <= k-max.')
    if args.output is None:
        if args.teste_manual_full:
            name = 'atividade_4_teste_manual_full'
        elif args.teste_manual:
            name = 'atividade_4_teste_manual'
        elif args.k_automatico:
            name = f'atividade_4_curvelet_auto_{args.k_min}_{args.k_max}'
        elif kc == kd:
            name = f'atividade_4_k{kc}'
        else:
            name = f'atividade_4_k{kc}_dino_k{kd}'
        args.output = ROOT / 'processed' / name
    if args.somente_visualizacoes:
        visualizations(args.output, pd.read_csv(args.output / 'campos.csv'),
                       pd.read_csv(args.output / 'patches.csv'))
        return
    fields, excluded = acquisitions(read_csv(ROOT / 'metadata/foundation_images.csv'))
    tables = {name: [] for name in ('campos','pares','objetos','patches')}
    tables['exclusoes'] = excluded
    features, previews = [], []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(device)
    if manual:
        manual_pair_test(fields, model, device, args.min_area, args.output, full=args.teste_manual_full)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    for folder in ('mascaras', 'sobreposicoes'):
        (args.output / folder).mkdir(exist_ok=True)
    for i, field in enumerate(fields, 1):
        analyze(field, model, device, args.min_area, tables, features, previews, args.output)
        if i % 10 == 0:
            print(f'{i}/{len(fields)} aquisições', flush=True)
    with threadpool_limits(limits=2):
        candidates = range(args.k_min, args.k_max+1) if args.k_automatico else None
        results = cluster(tables, features, previews, args.output, kc, k_dino=kd, candidates=candidates)
    for name in ('campos','pares','exclusoes'):
        results[name] = pd.DataFrame(tables[name])
    results['walleriana'] = walleriana(results['campos'])
    for name, frame in results.items():
        frame.to_csv(args.output / (name + '.csv'), index=False)
    compare_examples(args.output, results['perfis'], results['patches'])
    visualizations(args.output, results['campos'], results['patches'])
    print(f'{len(results["objetos"])} objetos em {args.output}')


if __name__ == '__main__':
    main()
