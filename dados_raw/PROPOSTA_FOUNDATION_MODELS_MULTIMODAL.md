# Proposta: identificação e quantificação multimodal de estruturas biológicas

## 1. Título provisório

**Foundation models e métodos clássicos para segmentação, representação e
quantificação de estruturas em microscopia óptica não linear multimodal**

## 2. Objetivo

Desenvolver e avaliar um pipeline que use modelos fundacionais de visão para
identificar e quantificar estruturas nas imagens SHG, TPEF, CARS e SRS,
primeiro em cada modalidade separadamente e depois pela integração dos canais.
Os estudos iniciais serão os conjuntos **BOCA** e **NERVO ÓPTICO – DEGENERAÇÃO
WALLERIANA**.

Os objetos de interesse iniciais são:

- fibras e feixes de colágeno em SHG;
- estruturas fibrilares autofluorescentes compatíveis com elastina em TPEF;
- regiões de contraste observadas nos canais CARS e SRS;
- regiões, objetos ou fenótipos multimodais definidos pela combinação espacial
  e pelos embeddings dos quatro contrastes.

O uso de “compatível com elastina” é deliberado. TPEF não é exclusivo de
elastina: outros fluoróforos endógenos ou corantes podem contribuir. A
atribuição precisa ser validada por morfologia, espectro, histologia ou marcação
de referência. Há evidência experimental de que SHG destaca colágeno fibrilar
e de que a autofluorescência de dois fótons pode destacar fibras elásticas,
mas a contribuição depende do tecido e da janela espectral
([Zhuo et al., 2009](https://pubmed.ncbi.nlm.nih.gov/19832952/);
[Zipfel et al., 2003](https://pmc.ncbi.nlm.nih.gov/articles/PMC165832/)).

## 3. O que cada modalidade mede

### SHG

SHG é um processo óptico coerente de segunda ordem. Ele é especialmente
sensível a estruturas não centrossimétricas e organizadas, como colágeno
fibrilar. O canal informa principalmente a distribuição, orientação e grau de
organização das fibras; sua intensidade também depende de polarização,
orientação, empacotamento e condições instrumentais. Portanto, intensidade SHG
não deve ser interpretada diretamente como concentração absoluta de colágeno.

### TPEF

TPEF mede fluorescência excitada pela absorção simultânea de dois fótons. Em
tecido sem marcação, pode registrar elastina e outros fluoróforos endógenos,
como NAD(P)H, FAD, queratina e produtos de ligação cruzada. Em BOCA, a hipótese
operacional será que parte das estruturas finas TPEF corresponde a elastina,
mas isso deverá ser validado. No subconjunto `24-02-2025/CORADO`, o corante
pode modificar ou dominar o contraste, razão pela qual `CORADO` e `SEM CORAR`
devem ser analisados separadamente.

### CARS e SRS

CARS e SRS são modalidades de **microscopia Raman coerente**. Neste projeto,
porém, os dados disponíveis são imagens prontas, e não espectros Raman. Por
isso, CARS e SRS serão tratados operacionalmente como dois canais de imagem
que mostram contrastes químicos e morfológicos complementares. Não será
necessário calcular deslocamento Raman para executar segmentação,
clusterização ou fusão.

- **CARS** detecta luz anti-Stokes gerada coerentemente. Produz sinal intenso,
  mas possui contribuição de fundo não ressonante e relação não linear mais
  complexa entre intensidade e concentração.
- **SRS** detecta a perda no pump ou o ganho no Stokes. Em geral, possui resposta
  mais diretamente proporcional à concentração e não sofre o mesmo fundo não
  ressonante característico do CARS.

Em aplicações biológicas, esses contrastes são frequentemente sensíveis a
regiões ricas em lipídios ou proteínas, especialmente quando a aquisição foi
configurada na região C–H
([revisão de SRS](https://www.mdpi.com/2304-6732/11/6/489);
[visão geral de CARS](https://pmc.ncbi.nlm.nih.gov/articles/PMC7098185/)). Neste
banco, entretanto, a composição molecular exata não será inferida somente pelo
nome `811` ou `816`. A análise proposta mede os padrões presentes nas imagens;
uma atribuição molecular mais específica poderá ser adicionada posteriormente
caso a configuração óptica completa da aquisição esteja disponível.

### Convenção dos detectores

Para as aquisições simultâneas `SRS-CARS`, esta proposta adota:

- **PMT1 / C0 = CARS**;
- **PMT2 / C1 = SRS**.

Essa convenção foi informada para o equipamento e substitui a interpretação
anterior do código. Antes das análises, ela deverá ser confirmada usando a
configuração óptica original ou uma amostra-controle com resposta conhecida.

## 4. Descrição dos dois conjuntos de dados

### 4.1 BOCA

O inventário atual contém **164 imagens**: 35 SHG, 30 TPEF, 48 CARS, 48 SRS e
3 arquivos ainda não classificados.

- `11-01-2025` contém SHG, TPEF e pares simultâneos CARS/SRS, com variações de
  comprimento de onda (`811`/`816` nos nomes), potência e número de aquisições.
- Nos pares CARS/SRS, C0 e C1 pertencem ao mesmo campo e podem ser tratados como
  co-registrados após uma verificação de alinhamento.
- SHG e TPEF foram adquiridos separadamente. Números de aquisição semelhantes
  não provam que representam exatamente o mesmo campo de visão.
- `24-02-2025` contém principalmente pares nominais SHG/TPEF, separados entre
  `CORADO` e `SEM CORAR`. Esses dois grupos não devem ser misturados sem incluir
  a condição de coloração como covariável.

Interpretação biológica inicial:

- SHG: rede de colágeno da matriz extracelular;
- TPEF: elastina candidata e outros fluoróforos do epitélio/matriz;
- CARS/SRS: contraste químico e morfológico complementar, analisado inicialmente
  sem impor uma classe molecular; regiões lipídicas ou proteicas serão hipóteses
  biológicas a validar, não rótulos automáticos.

### 4.2 NERVO ÓPTICO – DEGENERAÇÃO WALLERIANA

O inventário atual contém **176 imagens**: 17 TPEF, 137 CARS, 8 SRS, 14 não
classificadas e nenhuma imagem nomeada como SHG. Existem diretórios associados
a `NO`, `retina`, `control`, `14` e `7/C1L7N`. Esses nomes sugerem regiões e
condições experimentais, mas sua semântica precisa ser confirmada com o
responsável pela aquisição antes de serem usados como rótulos biológicos.

Interpretação inicial:

- CARS/SRS: alterações no contraste químico e na morfologia do tecido. Uma
  hipótese relevante é que parte do contraste acompanhe a mielina e sua
  fragmentação/remoção durante a degeneração, mas a imagem será quantificada
  antes de receber essa interpretação;
- TPEF: autofluorescência tecidual, que não deve ser chamada automaticamente de
  elastina no nervo óptico.

O conjunto Walleriana é adequado para estudar alterações químicas e
morfológicas, mas não contém todas as quatro modalidades. BOCA permanece o
conjunto principal para a demonstração multimodal completa.

## 5. Estratégia computacional

### Modelos fundacionais

- **micro-sam/SAM**: propostas de máscara, anotação assistida e posterior
  fine-tuning com poucas máscaras revisadas por especialista;
- **Cellpose-SAM**: baseline de instâncias, sobretudo para estruturas celulares;
- **DINOv2 ou Cell-DINO**: embeddings globais, por patch e por objeto para
  clusterização e recuperação de imagens semelhantes.

SAM e Cellpose não atribuem automaticamente significado biológico às máscaras.
A classe “colágeno”, “elastina” ou “domínio lipídico” deve vir da modalidade,
de anotação ou de validação experimental.

### Baselines clássicos

- **Curvelet Transform** para redução de ruído e realce de estruturas curvas e
  direcionais;
- **CT-FIRE** como referência para extração e quantificação de fibras
  individuais;
- Sato/Frangi multiescala para estruturas finas;
- Otsu ou limiar adaptativo;
- morfologia matemática e remoção de componentes pequenos;
- esqueleto, transformada de distância e rastreamento de fibras;
- watershed/componentes conexos para separar objetos.

### Quantificações

- número e área de objetos;
- fração de área positiva;
- comprimento, largura, orientação, tortuosidade e ramificações de fibras;
- intensidade média/mediana por objeto em cada modalidade;
- colocalização e distância entre objetos multimodais;
- embedding, cluster e incerteza por imagem/objeto.

### Como será feita a fusão multimodal

A fusão somente será executada quando as imagens representarem o mesmo campo de
visão. Os canais CARS e SRS da mesma aquisição são pares diretos. Para combinar
SHG e TPEF com eles, será necessário confirmar o pareamento e registrar as
imagens. Se não houver correspondência espacial, será possível comparar os
resultados por grupo, mas não fazer fusão pixel a pixel.

O procedimento será:

1. **Pareamento:** selecionar imagens do mesmo tecido, condição e campo de
   visão. Pares duvidosos ficam fora da análise multimodal.
2. **Registro:** usar um canal como referência e alinhar os demais por
   translação/rotação; quando necessário, usar registro afim ou deformável. A
   qualidade será verificada pela sobreposição de estruturas compartilhadas.
3. **Normalização por canal:** aplicar percentis robustos separadamente em SHG,
   TPEF, CARS e SRS, evitando que o canal de maior intensidade domine a fusão.
4. **Segmentação individual:** produzir uma máscara por modalidade usando o
   foundation model e o melhor método clássico. Essa etapa permanece como
   baseline obrigatório.
5. **Fusão multimodal:** avaliar três abordagens complementares descritas a
   seguir.

#### A. Fusão precoce, antes da segmentação

As modalidades registradas serão combinadas em uma imagem composta fornecida
ao micro-sam. Para duas modalidades, os canais RGB serão `[A, B, média(A,B)]`.
Para três, cada modalidade ocupará um canal. Para quatro modalidades, serão
testadas duas opções: projeção normalizada de quatro para três canais por PCA e
um pequeno adaptador treinável `4 → 3` antes do encoder. O modelo produzirá uma
segmentação baseada simultaneamente nos contrastes presentes.

Vantagem: o modelo pode usar contornos fracos em um canal que aparecem mais
claros em outro. Limitação: a composição RGB é artificial e pode ocultar a
origem física de cada sinal.

#### B. Fusão de características

Cada modalidade passará separadamente pelo DINOv2/Cell-DINO. Para uma imagem ou
objeto serão obtidos quatro vetores: `z_SHG`, `z_TPEF`, `z_CARS` e `z_SRS`.
Depois de normalizados, eles serão concatenados:

```text
z_multimodal = [z_SHG | z_TPEF | z_CARS | z_SRS]
```

O vetor multimodal será usado para clusterização, classificação e recuperação
de imagens semelhantes. Também será testada a média ponderada dos embeddings,
com pesos aprendidos apenas no conjunto de treino.

Vantagem: preserva uma representação própria para cada modalidade e aceita
canais com aparências muito diferentes. Essa será a abordagem principal para
clusterização.

#### C. Fusão tardia por objetos e máscaras

As máscaras individuais não serão simplesmente somadas, porque colágeno,
elastina candidata e regiões CARS/SRS não representam a mesma estrutura. Em vez
disso, será construída uma tabela multimodal por objeto. Por exemplo, para cada
fibra SHG serão calculados:

- comprimento, largura, orientação e tortuosidade em SHG;
- intensidade e sobreposição do TPEF ao redor da fibra;
- intensidade média de CARS e SRS dentro e ao redor da fibra;
- distância até o objeto TPEF, CARS ou SRS mais próximo;
- embedding de cada modalidade na região correspondente.

Esses atributos serão unidos em uma linha por objeto e usados para criar
clusters multimodais interpretáveis. Quando duas máscaras representarem o
mesmo tipo de objeto, serão testados consenso por interseção, união e votação.

#### Comparação com as modalidades individuais

Serão executados os mesmos experimentos com:

- somente SHG;
- somente TPEF;
- somente CARS;
- somente SRS;
- CARS + SRS;
- SHG + TPEF;
- todas as modalidades disponíveis e corretamente pareadas.

Para segmentação, a comparação usará Dice, IoU, precisão, revocação, erro na
contagem e erro nas medidas de comprimento/largura. Para clusterização, serão
usados silhouette score, estabilidade entre reamostragens e associação com os
grupos biológicos conhecidos. Um estudo de ablação retirará um canal por vez.
Assim será possível dizer não apenas se a fusão melhora o resultado, mas qual
modalidade fornece a melhoria.

## 6. Cinco atividades propostas

### Atividade 1 — Auditoria dos canais e pequeno padrão-ouro

Selecionar 10–20 campos de BOCA, confirmar PMT1/PMT2, identificar quais imagens
correspondem ao mesmo campo, realizar o registro espacial e anotar manualmente
algumas regiões de colágeno, elastina candidata e fundo. Entrega: manifesto de
pareamento, imagens registradas e pequeno conjunto de teste. Não é necessária
uma análise espectral para esta atividade.

### Atividade 2 — Colágeno SHG: foundation model versus algoritmo clássico

Aplicar micro-sam e Cellpose-SAM às imagens SHG. Comparar com (a) Curvelet
Transform + FIRE/CT-FIRE e (b) Sato + Otsu + morfologia + esqueletização. Medir
Dice/IoU, cobertura, comprimento, largura e orientação das fibras. Entrega:
tabela comparativa e overlays revisáveis.

### Atividade 3 — Estruturas TPEF e hipótese de elastina

Aplicar micro-sam às imagens TPEF de BOCA e comparar com Curvelet Transform,
Frangi/Sato e limiar adaptativo. Comparar `CORADO` e `SEM CORAR`, e medir
colocalização/distância em relação ao colágeno SHG quando houver registro.
Entrega: máscaras e análise das fontes de falso positivo; o resultado será
chamado “elastina” somente após validação.

### Atividade 4 — Fenótipos químicos em CARS/SRS

Nos pares simultâneos, comparar máscaras e intensidade de CARS e SRS. Usar
DINOv2/Cell-DINO para agrupar patches ou objetos e comparar com um baseline
clássico de Curvelet/filtragem, limiar e componentes conexos. Em BOCA, descrever
os padrões CARS/SRS sem forçar uma atribuição molecular. Em Walleriana, testar
a sensibilidade a alterações compatíveis com desorganização de mielina.
Entrega: clusters interpretáveis e medidas por objeto.

### Atividade 5 — Fusão multimodal e estudo de ablação

Executar a fusão precoce, a concatenação de embeddings e a tabela multimodal
por objeto descritas acima. Comparar cada resultado com SHG, TPEF, CARS e SRS
individuais, além dos pares CARS+SRS e SHG+TPEF. Retirar uma modalidade por vez
para medir sua contribuição. Entrega: máscaras, clusters, tabela de ablação,
ranking das estratégias e pipeline reproduzível para BOCA, seguido de
validação parcial em Walleriana.

## 7. Critérios de sucesso e riscos

Critérios mínimos:

- melhoria sobre o baseline clássico ou redução relevante do esforço de
  anotação;
- Dice/IoU reportados em imagens não usadas para ajuste;
- repetibilidade das métricas de objetos;
- clusters estáveis e não explicados apenas por potência, ganho, resolução ou
  número de aquisições;
- resultados separados por condição experimental e modalidade.

Principais riscos:

- domínio das imagens diferente dos dados de treinamento dos foundation models;
- confusão entre objeto visual e entidade biológica;
- ausência de co-registro entre modalidades adquiridas separadamente;
- atribuição biológica incorreta de padrões CARS/SRS sem validação externa;
- vazamento entre treino e teste quando imagens do mesmo campo aparecem em
  diferentes configurações de potência ou número de aquisições.

## 8. Resultado esperado

Um pipeline com inventário, pareamento, segmentação assistida, baselines
clássicos, embeddings, clusterização e métricas por objeto, acompanhado de um
pequeno conjunto anotado e de uma comparação objetiva entre modelos
fundacionais e processamento clássico.
