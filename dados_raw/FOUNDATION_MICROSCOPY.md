# Segmentação e clusterização multimodal com foundation models

Este protótipo indexa as imagens em `raw/IMAGENS`, segmenta cada modalidade
separadamente com **micro-sam**, extrai representações visuais com **DINOv2** e
permite comparar clusters unimodais com clusters multimodais.

## Hipótese experimental

1. Segmentar SHG, TPEF, CARS e SRS separadamente.
2. Extrair um embedding por imagem e comparar a estrutura dos clusters entre
   modalidades.
3. Para aquisições pareadas, concatenar os embeddings na ordem das modalidades
   e repetir a clusterização.
4. Só fundir máscaras pixel a pixel depois de confirmar co-registro. O manifesto
   marca os dois canais de uma mesma pasta `SRS-CARS` como `exact`; os demais
   pareamentos inferidos do nome são `candidate`.

SAM/micro-sam produz propostas de objetos, não classes biológicas. Uma pequena
rodada de anotação e fine-tuning será necessária para transformar as máscaras
em classes como colágeno, célula, lipídio ou fundo. DINOv2 agrupa sem rótulos;
os clusters também precisam de interpretação posterior.

## Preparação

```bash
cd /home/clovis/lioas_dbv2
python -m venv .foundation-env
.foundation-env/bin/pip install -r requirements-foundation.txt
```

Instale `micro-sam` em um ambiente isolado e informe seu Python com
`--microsam-python`. A primeira execução do DINOv2 baixa pesos pelo PyTorch Hub.

## 1. Inventário

```bash
python -m foundation_microscopy.cli inventory
```

O comando cria `metadata/foundation_images.csv` e imprime as modalidades de
cada grupo. Em `SRS-CARS`, **PMT1/C0 é interpretado como CARS e PMT2/C1 como
SRS**, conforme a convenção experimental informada para este equipamento. Essa
atribuição deve ser confirmada com uma aquisição-controle e a configuração dos
detectores antes da análise definitiva.

## 2. Segmentação separada

Teste rápido, sem foundation model:

```bash
python -m foundation_microscopy.cli segment \
  --group BOCA --backend classical --limit 8
```

Execução com micro-sam:

```bash
python -m foundation_microscopy.cli segment \
  --group BOCA --backend microsam \
  --microsam-python /caminho/do/ambiente/bin/python
```

As máscaras são gravadas por `grupo/modalidade`, o que permite avaliar SHG,
TPEF, CARS e SRS de forma independente.

Para segmentar conjuntamente os canais SRS e CARS simultâneos:

```bash
python -m foundation_microscopy.cli segment-fused \
  --group BOCA --modalities SRS,CARS --backend microsam \
  --microsam-python /caminho/do/ambiente/bin/python
```

O comando monta uma composição RGB e, por padrão, aceita apenas pares `exact`.
Com três modalidades cada canal ocupa R/G/B; com quatro, SHG e TPEF ficam em
canais próprios e CARS/SRS são combinados no terceiro canal. Essa projeção deve
ser comparada com fusão tardia e com um encoder adaptado a número arbitrário de
canais em uma etapa posterior.

## 3. Embeddings e clusters unimodais

```bash
python -m foundation_microscopy.cli embed --group BOCA
python -m foundation_microscopy.cli cluster \
  --input processed/foundation_microscopy/embeddings.npz \
  --output processed/foundation_microscopy/clusters_por_modalidade.csv \
  --clusters 5 --by-modality
```

## 4. Fusão multimodal

Para SRS+CARS simultâneos, use apenas pareamentos exatos:

```bash
python -m foundation_microscopy.cli fuse \
  --input processed/foundation_microscopy/embeddings.npz \
  --output processed/foundation_microscopy/srs_cars_fused.npz \
  --modalities SRS,CARS --exact-only

python -m foundation_microscopy.cli cluster \
  --input processed/foundation_microscopy/srs_cars_fused.npz \
  --output processed/foundation_microscopy/clusters_srs_cars.csv \
  --clusters 5
```

Para SHG+TPEF+CARS+SRS, primeiro revise manualmente os grupos `candidate` do
manifesto e confirme campo de visão, resolução e registro. Não trate igualdade
de número de aquisição como prova de co-registro.

## Avaliação recomendada

- Segmentação: Dice/IoU em um conjunto pequeno anotado por modalidade.
- Estabilidade: concordância dos clusters entre seeds e reamostragens.
- Biologia: associação dos clusters com tecido, condição, potência, ganho e
  número de aquisições; esses fatores técnicos podem dominar o embedding.
- Validação multimodal: comparar concatenação, média e ablação de cada canal.

