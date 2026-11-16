# Individual identification of pasture cattle from UAV imagery: a controlled protocol for separating coat-pattern purification from backbone choice

<!-- =================================================================== -->
<!-- DRAFT STATUS — remove this block before submission.                 -->
<!-- Every numeric result below that depends on a training run is marked -->
<!-- 【待补: ...】. No retrieval figure in this draft is a measurement.    -->
<!-- Provenance for every number that IS real is given in an HTML comment -->
<!-- immediately beside the number. Full audit: paper/evidence-audit.md. -->
<!-- =================================================================== -->

**Authors**
Tianyi Ma^a,^\*

^a^ School of Computer and Artificial Intelligence, Zhengzhou University, Zhengzhou 450001, China

\* Corresponding author. E-mail: 【待补: 通讯作者邮箱】; ORCID: 0000-0003-1167-621X

---

## Abstract

Individual identification is the enabling step for precision-livestock applications that treat an
animal rather than a herd as the unit of management: growth monitoring, individual feed
conversion, disease surveillance and provenance records. In pasture systems, ear tags, collars and
boluses carry recurring cost and require handling, which motivates identification from aerial
imagery already collected by unmanned aerial vehicles (UAVs). The dorsal coat pattern is a usable
biometric from above, but reported accuracies are hard to compare because studies differ
simultaneously in segmentation, crop policy, backbone and gallery composition. We address this
comparability problem with a controlled factorial protocol rather than a new architecture. Two
paired crops are derived from the same detections in the same frames: a mask-purified dorsal coat
region, with pasture, shadow and limbs excluded, and a plain bounding-box crop in which they
remain. Three embedding networks — a compact convolutional trunk, a Vision Transformer and
ResNet-50 — are trained under one shared additive angular margin objective with identical
resolution, embedding dimension, optimiser and epoch budget, and evaluated by cosine retrieval on
one frozen identity-disjoint split. The 2 × 3 design isolates mask-based purification from
backbone capacity, and reports parameters and multiply–accumulates for the inference network
alone, excluding the training classifier. We describe the protocol and its reference
implementation, and report each backbone's budget: 1.83 M, 5.62 M and 24.56 M parameters at 242 M,
1075 M and 4088 M multiply–accumulates per 224 × 224 image. Separating crop purification from
backbone capacity is, we argue, a precondition for interpreting aerial cattle re-identification on
a working farm. 【待补: 六单元格检索结果一句话摘要（含最佳单元格 Rank-1 与 Δ_SAM 方向），
在实测完成后补入，不得以占位数值代替】

**Keywords:** cattle re-identification; unmanned aerial vehicle; precision livestock farming;
coat pattern; metric learning; background suppression

---

## 1. Introduction

Precision livestock farming has moved, over the past decade, from herd-level sensing towards
management at the level of the individual animal. The reason is practical rather than
methodological. A growth curve, a feed-conversion ratio, a lameness score or a treatment record
is only interpretable if it is attached to a known animal, and the value of attaching it grows
with the length of the record: an animal whose weight is measured weekly for a full finishing
period supports decisions that a single liveweight measurement cannot. Once individual records
exist, they propagate into breeding-value estimation, into the detection of animals that
diverge from their own trajectory, and into the traceability documentation that processors and
regulators increasingly require.

All of this presupposes that the animal can be recognised, repeatedly, without being handled.
The identifiers in routine use each carry a cost that is easy to underestimate at the scale of a
commercial herd. Ear tags are inexpensive per unit but are lost at a rate that is non-trivial
over a finishing cycle, and reading them requires the animal to pass a reader or to be
restrained. Radio-frequency identification in a rumen bolus or an ear tag solves the reading
problem for fixed infrastructure — a crush, a weighing platform, an automatic milking system —
but does nothing for an animal standing in a paddock. Collars and accelerometer tags provide
continuous data at the cost of a device per animal, a battery per device, and a handling event
to fit and to recover it. None of these options is unreasonable; each is simply expensive in a
currency that pasture-based systems are short of, which is contact with the animal.

Imagery collected by unmanned aerial vehicles offers a different trade. A single flight can
cover a paddock in minutes and yields a few hundred to a few thousand frames at a ground
sampling distance of a centimetre or two, and the aircraft never touches the animal, so the
measurement does not perturb the behaviour it is meant to observe. The same flight that
supports a body-condition or liveweight estimate can, in principle, also establish which animal
is being measured. Wang et al. (2023) review how three-dimensional and imaging approaches have
been used for cattle growth management in this setting, and Yan et al. (2026) demonstrate a UAV
framework for morphological and growth analysis of feedlot cattle. The missing link in that
chain is identity.

That cattle can be identified from above is now established. The dorsal coat pattern of
Holstein-Friesian and Simmental animals is sufficiently individual, and sufficiently stable over
weeks, to support retrieval. Andrew et al. (2017) showed visual localisation and individual
identification of Holstein-Friesian cattle from a moving platform, and Andrew et al. (2021)
developed the approach into a deep metric learning system with an explicit treatment of the
open-set case. Wang et al. (2025) extended coat-pattern identification to aerial and ground
imagery of 96 Simmental beef cattle and proposed it as a best practice, releasing both the
imagery (Wang, 2024) and the accompanying code and logs. Related biometrics have been pursued in
parallel: muzzle patterns (Li et al., 2022), gait dynamics combined with coat pattern (Chen et
al., 2026), and multi-modal fusion of several appearance cues (Pathak and Prakash, 2025).

What has not been settled is how to read these numbers. Reported rank-1 accuracies for cattle
re-identification range from the mid-eighties to above ninety-nine percent, and the spread is
not primarily a difference in recognition architectures. It reflects differences in what the
network is allowed to see. A crop that includes a section of fence, a feed trough, a pen wall or
a distinctive patch of bare ground gives the network a shortcut that correlates with identity in
a single-farm dataset and will not transfer to the next farm. A crop that has been masked down to
the coat removes that shortcut but also removes pixels that a network might legitimately have
used, and it introduces its own dependence on the quality of the mask. When study A reports a
masked-crop accuracy and study B reports a bounding-box accuracy on a different farm, the
comparison between them is uninformative, because crop policy and architecture vary together.

This paper treats that confound as the object of study. We fix the detection, fix the identity
split, fix the training objective and fix the optimiser, and then vary two factors
independently: the crop policy, with two levels, and the recognition backbone, with three. The
resulting six cells form a 2 × 3 design in which the difference between the two crops for a
given backbone is attributable to the crop, and the difference between backbones for a given
crop is attributable to the network. Neither difference is contaminated by the other, and
neither is contaminated by a change in the training loss.

The contributions are as follows.

First, we specify a paired-crop protocol in which the two experimental arms are derived from the
same detections in the same source frames, so that they differ in retained pixels and not in the
list of animals, the number of images, or the identity assignment. The protocol defines the
mask-purified arm as the region inside a segmentation mask of the dorsal coat and the control
arm as the axis-aligned detector box, and it forbids any re-selection of images between arms.

Second, we place three architecturally distinct embedding networks under one additive angular
margin objective with a single set of scale, margin and embedding-dimension values, so that the
loss cannot act as a hidden variable in the architecture comparison. This is a stricter
condition than is usual in the cattle re-identification literature, where each published
backbone tends to arrive with its own mining objective.

Third, we define an efficiency accounting that is meaningful for deployment on an aircraft or
on an edge device: parameters and multiply–accumulates are counted on the inference embedding
network at the common input resolution, with the training classifier excluded, because the
class-weight matrix scales with the number of enrolled animals and is discarded at test time.
Reporting it that way prevents a backbone from appearing cheaper or more expensive than it is
merely because the study happened to enrol more animals.

Fourth, we release a reference implementation that enforces the protocol, including the retrieval
metric, the paired-crop manifests and the cost accounting, so that the comparison can be repeated
on other farms and other breeds without re-deriving the conditions.

The remainder of the paper is organised as follows. Section 2 reviews individual identification
in livestock, the use of aerial imagery in pasture management, and the metric-learning losses on
which the shared objective is based. Section 3 gives the materials and methods, including data
acquisition and annotation, the crop construction, the three backbones, the objective, and the
experimental setup. Section 4 reports retrieval performance and computational cost for the six
cells. Section 5 discusses what the design implies for on-farm use, where it can fail, and what
would be required before an identification system of this kind could be trusted as a management
record. Section 6 concludes.

## 2. Related work

### 2.1 Individual identification in livestock

Identification of individual animals has a long history in animal science, and the biometric
approaches that dominate the computer-vision literature are relatively recent arrivals. The
taxonomy that matters for the present work is between identifiers that require an artefact and
identifiers that read the animal itself. Artefact-based systems — ear tags, collars, boluses,
brands — are mature, cheap per reading, and well understood operationally; their weaknesses are
loss, damage, and the handling event required to restore them. Biometric systems read a property
of the animal, and therefore cannot be lost, but they require a sensing modality and a model, and
their failure modes are statistical rather than mechanical.

Within biometrics, cattle offer an unusual number of usable cues. The muzzle print is the
closest analogue to a human fingerprint and has been approached with classical texture
descriptors and with convolutional networks; Li, G. et al. (2022) identified beef cattle from
muzzle images, Li, Z. et al. (2022) proposed a lightweight model for cattle face recognition
aimed at practical deployment, and muzzle-based identification continues to be an active line. The dorsal coat pattern is the cue of choice for overhead imagery, because
it is the surface that a downward-looking camera actually sees. Gait and locomotion carry
identity information as well, and Chen et al. (2026) combined gait dynamics with coat pattern in
a multimodal framework for Holstein cattle, reporting that the two cues are complementary rather
than redundant. Pathak and Prakash (2025) pursued attention-based fusion of multiple modalities
for the same end. The lesson common to these studies is that no single cue is uniformly
available: a cow lying down presents a coat pattern but no gait, and a cow in a narrow crush
presents a muzzle but no dorsal view. A management system therefore benefits from a cue that is
available in the conditions the aircraft actually encounters, which for pasture surveillance is
the back.

The identification task itself comes in two forms that are often conflated. Closed-set
identification asks which of a fixed enrolled set an image belongs to, and is evaluated by
classification accuracy over that set. Open-set re-identification asks whether a probe matches
any enrolled identity at all, and must therefore support a rejection decision; Andrew et al.
(2021) treat this explicitly, and Kuncheva (2021) formalises the restricted-set-classification
view in which the candidate set for each probe is constrained by context. The problem is not
merely taxonomic: Shumaly et al. (2025) show that uncertain identification propagates into
phenotypic analysis, so a misassigned image does not simply drop out of the dataset but corrupts
the trait estimate it was collected to support. The distinction has practical weight, because a system that must reject unknown animals — a newly purchased group,
or a neighbouring herd visible from the same flight line — is a harder problem than one that
assumes every probe is enrolled. Retrieval metrics of the rank-k family, which report whether
the true identity appears among the k nearest gallery identities, belong to the closed-set
family unless a rejection threshold is added. We report rank-k and state the closed-set
assumption explicitly.

### 2.2 Aerial and UAV-based sensing for pasture livestock

The use of UAVs in livestock systems has grown with the availability of lightweight,
high-resolution RGB payloads. The applications cluster into counting and localisation, body
measurement, behaviour monitoring, and health surveillance; the breadth of this literature is
surveyed by Rohan et al. (2024) for behaviour recognition and by Rahman et al. (2025) for deep
learning in animal farming more generally. Counting is the most mature: Tian et
al. (2019) demonstrated automated pig counting with deep networks, and the same detection
machinery transfers to cattle with modest adaptation, building on the single-stage detection
formulation of Redmon et al. (2016) and its later refinements such as YOLOv9 (Wang et al., 2024).
Detection-driven pipelines in this area also need to estimate intake and intake behaviour, for
which Bezen et al. (2020) combined an RGB-D camera with deep networks to measure individual cow
feed intake — an example of the per-animal records that only become possible once identity is
resolved. Body measurement from above is the
application with the clearest economic pull, because liveweight and body condition drive both
management and breeding decisions, and because an aerial estimate avoids the crush. Yan et al.
(2026) report a UAV framework for morphological and growth analysis of feedlot cattle, and the
dataset of Wang (2024) was collected with exactly this application in mind, pairing UAV-based
LiDAR with RGB so that three-dimensional shape and two-dimensional appearance are available
together.

Wang et al. (2023) survey the three-dimensional computer-vision methods used for cattle growth
management and note a recurring practical constraint: the animals move, the aircraft moves, and
the resulting imagery is neither uniformly illuminated nor uniformly oriented. For
identification this matters more than for measurement, because a measurement can be aggregated
over many frames while an identity decision has to be made per animal per frame or per short
track. Detection and tracking are therefore prerequisites rather than refinements. Han et al.
(2023) address multi-cattle tracking in crowded pens, Liu et al. (2024) survey deep-learning
tracking for animals, and Zhao et al. (2026) propose a spatio-temporal identity system that
maintains identity across cameras over long periods. These systems and the present work are
complementary: tracking supplies the association across frames, and re-identification supplies
the association to an enrolled identity when a track begins or when the animal re-enters after
an absence.

Two constraints recur across this literature and shape our design. The first is that pen
infrastructure is a nuisance variable of exactly the kind that correlates with identity. Animals
are grouped by age, weight or treatment, so the pen a cow occupies is informative about which cow
it is. In a single-farm dataset this correlation can be strong enough that a network reading the
background outperforms a network reading the animal, which is a well-known failure in person
re-identification and applies with more force here because group-housed cattle are physically
confined to their group. The second is that the annotation budget is spent on identity labels,
not on segmentation masks, so any masking stage must operate without per-image mask supervision.
Both constraints point towards an automatic segmentation step interposed between detection and
recognition, and towards an experimental design that can measure whether that step helps.

### 2.3 Deep metric learning and margin-based objectives

Re-identification is naturally a metric-learning problem: the network maps an image to an
embedding, and identity decisions are made by comparing embeddings. The dominant family of
objectives in modern practice is margin-based classification over identity labels, which
sidesteps the combinatorial cost of tuple sampling while producing embeddings with the desired
geometry. Deng et al. (2019) introduced the additive angular margin loss, which places the
margin in angular space on the hypersphere after normalising both the embedding and the class
weights. With scale $s$ and margin $m$, the target logit becomes $s\cos(\theta_y + m)$ while the
non-target logits remain $s\cos\theta_j$, and the softmax cross-entropy is applied on top. The
geometric reading is that the loss compacts each identity into a cone around its class centre
and pushes the cones apart by the angular margin, which is precisely the property retrieval
needs.

Two properties of this objective matter for the comparison we build. First, the class-weight
matrix exists only during training and is discarded at inference, so the exported model is an
embedding function and the cost accounting should reflect that. Second, and less often remarked,
the objective is invariant to the number of enrolled identities in a way that makes it a fair
common denominator: changing the backbone while holding $s$, $m$ and the embedding dimension
fixed changes the representation but not the geometry of the decision rule. A study in which each
backbone arrives with its own triplet-mining or contrastive objective cannot make that claim,
because a difference in accuracy may reflect the objective rather than the architecture.

The transformer line of work supplies the architectural alternative to convolutional trunks.
Dosovitskiy et al. (2021) showed that a pure transformer over image patches is competitive for
image recognition given sufficient data or adequate regularisation; Liu et al. (2021) reintroduced
hierarchical, locally-windowed attention that recovers the multi-scale behaviour of convolutional
networks; and He et al. (2021) adapted the transformer to re-identification, where the token
sequence accommodates part-based cues. For livestock identification the transformer's appeal is
that a coat pattern is a patch-structured signal, and global self-attention can relate a patch on
the flank to one on the shoulder across the whole animal. Its cost is quadratic attention over
the token grid, which is why structured pruning — removing whole heads or channels rather than
individual weights — is the natural compression route when the target is an edge device. Whether
a pruned transformer retains its advantage is an empirical question, and it is one that can only
be answered if the comparison to the unpruned model and to convolutional alternatives is run
under a common objective. That is the question our 2 × 3 design is built to answer.

Finally, the segmentation component. Kirillov et al. (2023) introduced the Segment Anything
Model, a promptable segmentation network trained on a very large mask corpus, which accepts a
box or point prompt and returns a mask for the object at that prompt. For livestock imagery this
is attractive because the detector already supplies the box, so no additional annotation is
needed to obtain a mask, and because a mask of the animal's dorsal region is exactly the
"purified crop" that the shortcut argument calls for. Whether the resulting purification helps
or hurts identification is not obvious a priori. It removes the pen context that may be acting
as a shortcut, which should help generalisation; it also removes pixels bordering the animal that
may carry genuine coat information at oblique view angles, which could hurt. The sign of the
effect is therefore an empirical outcome, and our protocol measures it per backbone rather than
assuming it.

## 3. Materials and methods

### 3.1 Study design

The experiment is a 2 × 3 factorial. The first factor is the crop policy applied between
detection and recognition, with two levels: a mask-purified crop derived from a segmentation of
the dorsal coat, denoted $I_{\mathrm{sam}}$, and a bounding-box crop of the original frame,
denoted $I_{\mathrm{box}}$. The second factor is the recognition backbone, with three levels: a
compact convolutional trunk, a Vision Transformer, and a residual network. Every one of the six
cells is trained from scratch on its own crop manifest and evaluated on the same identity split.
The quantity of interest for the first factor is the difference in rank-1 between the two crops
for a fixed backbone,

$$\Delta_{\mathrm{SAM}} = \mathrm{Rank\text{-}1}(I_{\mathrm{sam}}) - \mathrm{Rank\text{-}1}(I_{\mathrm{box}}),$$

<!-- implemented as rank1_gain() in src/evaluate.py:42-49 -->

which is computed by the implementation as a single function so that the sign convention cannot
drift between cells. A positive value means the mask raised rank-1; the sign is an outcome of the
experiment and is not assumed. The quantity of interest for the second factor is the ordering of
the three backbones within a crop arm, read together with their parameter and multiply–accumulate
counts.

The design is deliberately paired rather than merely balanced. Both arms are constructed from the
same detections in the same source frames, so the two manifests contain the same animals, the same
number of images per animal and the same assignment to training, gallery and query. An arm
therefore differs from its partner only in which pixels are retained. This is a stronger
condition than training two models on two independently assembled datasets, and it is what allows
$\Delta_{\mathrm{SAM}}$ to be interpreted as an effect of cropping rather than of sampling.

### 3.2 Animals, data acquisition and annotation

The imagery originates from the publicly deposited UAV cattle archive of Wang (2024), released
under CC BY 4.0 as a single archive (`dataShare.zip`, 26,486,342,645 bytes) and described in the
companion data article. The archive documents 96 Simmental beef cattle imaged from unmanned
aerial vehicles as part of a study on individual-level growth monitoring in precision livestock
farming, and it is accompanied by the code, results and logs of the coat-pattern identification
work of Wang et al. (2025).

<!-- Dataset identity verified against the Zenodo API during the audit:
     title, depositor (Wang, Yaowu), publication date 2024-08-23, licence cc-by-4.0,
     one file dataShare.zip of 26,486,342,645 bytes.
     Animal count (96 Simmental) corroborated by companion record 10.5281/zenodo.13361729:
     "96 Simmental male beef cattle in the datasets UAV-RGB and Ground-RGB"
     See paper/evidence-audit.md sections 6 and 12. -->

The acquisition parameters that a reader needs in order to judge transferability are the
following, and they must be taken from the archive documentation rather than restated from
memory: flight altitude and resulting ground sampling distance, camera model and focal length,
image dimensions in pixels, number of flights and their dates, the paddock or pen layout, and the
range of solar elevation and cloud cover under which frames were accepted.
【待补: 飞行高度与地面采样距离 (GSD)、相机型号与焦距、原始图像分辨率、飞行架次与日期、圈舍/放牧地布局、光照条件范围——须据 Zenodo 数据说明逐项填写】

Annotation is at the level of identity, not of pixels. Each image is associated with the ear-tag
identifier of the animal it depicts, and the identity table is the only supervision used. No
segmentation mask is annotated: the mask-purified arm is produced by prompting a segmentation
model with the detector box, so the cost of the masking stage is inference time rather than
labelling time. The identity assignment is partitioned per animal into training, gallery and
query subsets, and the partition is frozen before any model is fitted.

【待补: 标注流程细节——标注人员数量与培训、身份核对方式（是否与耳标/牧场记录交叉验证）、
疑难样本（遮挡、多牛同框、部分出画）的处理规则、标注一致性检查结果】

Table 1 accounts for the images and identities that enter each split. Both crop arms draw on the
same accounting, so the table has one version rather than one per arm.

| Quantity | Value |
| --- | ---: |
| Identities in the archive | 96 |
| Identities retained for evaluation | 【待补】 |
| Images in the training split | 【待补】 |
| Images in the gallery split | 【待补】 |
| Images in the query split | 【待补】 |
| Images per identity: mean / median / min / max | 【待补】 |
| Identities excluded, with reason | 【待补】 |

*Table 1. Data accounting for the frozen split. Identities are counted once; a query image and a
gallery image of the same animal are never two views exported from the same detection. The
retained-identity count may be below 96 if an animal lacks a qualified training image or a query
image, and any such exclusion must be reported here rather than silently absorbed.*

<!-- The 96 figure is a property of the source archive (verified against Zenodo record
     13364663 and its companion 13361729, which states "96 Simmental male beef cattle");
     it is not a count made by this repository. Every other cell in Table 2 is unknown
     because data/ was never committed and no run manifest exists. See
     paper/evidence-audit.md section 6. -->

Because the study involves only overhead observation of animals in their normal environment, with
no handling, restraint or intervention, the ethical position should be stated explicitly rather
than left implicit.
【待补: 动物伦理审批编号与所属机构、以及"仅为非接触式航拍观测、未对动物实施任何处置"的声明】

### 3.3 Crop construction

Detection precedes the fork. For every source frame a detector localises the animal and returns an
axis-aligned box $B$. That box is the only spatial prompt passed to the segmentation stage, and it
is shared by both arms, so the arms cannot differ in which animals were found.

In the purified arm the box prompts a segmentation model, which returns a binary mask $M$ of the
animal's dorsal region within the box. The pixels retained for recognition are

$$I_{\mathrm{sam}} = \operatorname{Crop}\left(I \odot M,\; B_M\right),$$

where $\odot$ is the element-wise product with $M$ broadcast over colour channels and $B_M$ is the
tight axis-aligned box of the foreground pixels of $M$. Pasture, hard shadow falling outside the
coat, and limbs that fall outside $M$ are therefore absent from $I_{\mathrm{sam}}$. In the control
arm the same box crops the original frame,

$$I_{\mathrm{box}} = \operatorname{Crop}(I, B),$$

and the segmentation model is not called, so ground, shadow and legs may remain inside the crop.

The detector and the segmenter are shared preprocessing. They are identical for all six cells and
for both arms, they are not trained as part of this study, and their cost is reported separately
from the recognition cost so that it cannot be attributed to one backbone.

<!-- Implementation status. The public reference implementation supplies this stage in two forms:
     (i) ingestion of pre-segmented per-identity folders (src/preprocess.py:37-93), and
     (ii) a detector-driven path (src/preprocess.py:110-180). In the committed revision the
     detector-driven path writes the plain box crop for both arms (src/preprocess.py:165-168),
     i.e. the mask is not applied. Any Δ_SAM reported from this code must therefore come from
     manifests produced by an external masking step. See paper/evidence-audit.md section 3. -->

### 3.4 Recognition backbones

Three embedding networks are compared. All three map a $224 \times 224 \times 3$ crop to a
512-dimensional embedding, which is then $\ell_2$-normalised, so that the inner product of two
embeddings is the cosine of the angle between them.

*Convolutional trunk.* The first backbone is a compact five-stage convolutional stack of
convolution–batch-normalisation–SiLU blocks with stride-2 downsampling, global average pooling,
and a linear projection to the embedding dimension. It is trained from random initialisation. It
serves as the low-capacity reference point: if a much larger network does not beat it under a
common objective and a common crop, then the extra capacity is not buying identification
performance on this data.

*Vision Transformer.* The second backbone is a patch-16 Vision Transformer with 12 blocks, 3
attention heads per block and a 192-dimensional token width, taking the class token as the pooled
representation and projecting it linearly to the embedding dimension. A Vision Transformer of this
size has global receptive field from the first block, which is the property that motivates its use
on a spatially extended pattern such as a coat.

*Residual network.* The third backbone is a ResNet-50 trunk with the classification layer removed
and a linear projection to the embedding dimension. It represents the capacity extreme of the
comparison and the architecture family most frequently used in the cattle-identification
literature.

<!-- All three are constructed in src/models.py:57-181 and were instantiated during the audit.
     Parameter and MAC figures in Table 4 were produced by calling the repository's own
     count_parameters() and estimate_flops() at 224x224 with configs/experiment.yaml. -->

Because the three backbones differ in capacity by more than an order of magnitude, comparing their
accuracy alone would be uninformative. Every cell is therefore reported with its parameter count
and its multiply–accumulate count, and the discussion reads accuracy against cost rather than
accuracy alone.

### 3.5 Shared training objective

All six cells are trained with the additive angular margin loss of Deng et al. (2019). Let
$z_i = \phi_\theta(x_i)/\lVert\phi_\theta(x_i)\rVert_2$ be the normalised embedding of sample $i$
and let $W_{y_i}$ be the normalised weight row of its true class. With $\theta_{y_i}$ the angle
between them, scale $s$ and angular margin $m$,

$$\ell_i = -\log \frac{\exp\left(s\cos(\theta_{y_i}+m)\right)}{\exp\left(s\cos(\theta_{y_i}+m)\right) + \sum_{j \neq y_i} \exp\left(s\cos\theta_j\right)}.$$

<!-- Implemented in src/arcface.py:44-58, including the cos(pi - m) guard for numerical
     stability of the margin on the non-target side. -->

The scale and the margin are shared constants, fixed at $s = 64$ and $m = 0.5$ for every cell and
every crop arm, and are taken from the ArcFace formulation rather than tuned per backbone.
<!-- configs/experiment.yaml:14-16 -->
Holding them fixed is what makes the architecture comparison interpretable: a change in accuracy
between backbones cannot be attributed to a change in the geometry of the decision rule.

At inference the class-weight matrix is discarded. Each network returns an $\ell_2$-normalised
embedding, and identities are ranked by cosine similarity. This is why the parameter and
multiply–accumulate counts in Section 3.7 exclude the classifier: its width is the number of
enrolled identities, which is a property of the deployment and not of the network.

### 3.6 Experimental setup

Training settings are shared across all six cells and are recorded in a single configuration file,
so that no cell can be tuned independently of the others.

| Setting | Value | Source |
| --- | --- | --- |
| Input resolution | 224 × 224 | `configs/experiment.yaml` |
| Embedding dimension | 512 | `configs/experiment.yaml` |
| Batch size | 8 | `configs/experiment.yaml` |
| Epoch budget | 30 | `configs/experiment.yaml` |
| Optimiser | Adam | `src/train.py:93` |
| Learning rate | 1 × 10⁻⁴ | `configs/experiment.yaml` |
| Weight decay | 1 × 10⁻⁴ | `configs/experiment.yaml` |
| ArcFace scale $s$ | 64.0 | `configs/experiment.yaml` |
| ArcFace margin $m$ | 0.5 | `configs/experiment.yaml` |
| Random seed | 0 | `configs/experiment.yaml` |
| Rank callback interval | every 5 epochs | `configs/experiment.yaml` |

*Table 2. Shared training configuration. Every value is read from one file and is identical for
all six cells; the table is a description of the protocol, not of an outcome.*

Augmentation is deliberately minimal, because aggressive augmentation would interact with the crop
factor: resizing to 224 × 224, random horizontal reflection with probability 0.5, colour jitter on
brightness, contrast and saturation at 0.15, 0.15 and 0.1, conversion to a tensor, and
normalisation with ImageNet statistics.
<!-- src/data.py:132-150 -->
No random cropping, rotation or erasing is applied, since each of these would partially undo the
crop policy that the experiment is designed to test. Horizontal reflection is admissible because
the dorsal coat pattern is not chiral, and because both arms receive it identically.

The training loop is a fixed-epoch procedure with no early stopping and no learning-rate schedule.
<!-- src/train.py:113-158 -->
A rank-1/rank-5 callback evaluates the current model on the gallery and query splits every five
epochs and at the final epoch, and the checkpoint with the best observed rank-1 is retained.
<!-- src/callbacks.py:19-52; src/train.py:137-156 -->
This is a model-selection step and is reported as such: it uses gallery and query data, and the
selection criterion is therefore not independent of the reported metric. It is applied identically
to all six cells, so it does not favour one cell over another, but it does mean that the reported
figures are best-observed rather than single-final-epoch values.

<!-- The epoch budget of 30 in configs/experiment.yaml should be reconciled with the schedule
     actually used for the reported runs before submission. The placeholder curve files in
     results/ describe a 1000-epoch schedule, which contradicts this configuration and is one of
     the reasons those files are excluded from this manuscript (paper/evidence-audit.md §1, §8). -->

【待补: 实际报告结果所用的训练轮次与是否使用学习率调度；若与上表不同，须给出最终采用的
配置并说明为何调整】

【待补: 训练硬件（GPU 型号、数量）、单细胞训练时长、总计算开销】

### 3.7 Evaluation protocol and cost accounting

Evaluation is closed-set retrieval. Each query image is embedded and ranked against the gallery
embeddings by cosine similarity, in descending order. Duplicate identities are suppressed during
ranking so that a query cannot be scored correct twice for the same animal, which matters whenever
a gallery contains more than one image per identity — the normal case here. Rank-$k$ is the
fraction of queries whose true identity appears among the $k$ top-ranked distinct identities.
<!-- src/evaluate.py:97-121 -->

We report rank-1 and rank-5. Rank-1 is the quantity a management system would act on when it makes
an automatic decision; rank-5 is the quantity that determines the usefulness of a short
human-verification list, which is a realistic operating mode for a system that proposes candidate
identities to a stockperson. Reporting both, rather than rank-1 alone, keeps that distinction
visible.

The gallery and query splits are frozen before any model is trained, and the two crop arms share
them exactly, so the comparison between arms is not confounded by a change in the evaluation set.
The number of gallery and query images is reported with the results.

<!-- src/evaluate.py:154-159, and src/data.py:18 which defines the three splits. Note that the
     repository defines no validation split; any hyperparameter chosen on gallery or query data
     must be disclosed as such. -->

Cost is measured on the inference embedding function only, at the common input resolution of
224 × 224, with single-image forward passes. Parameters are counted as trainable parameters of the
embedding network after any compression is applied. Multiply–accumulates are counted by
forward-hook accumulation over convolution and linear layers.
<!-- src/models.py:202-236 -->

Three remarks on units. The implementation returns multiply–accumulates, not floating-point
operations; the two differ by a factor of two and the distinction is stated wherever a figure is
given. The ArcFace class-weight matrix is excluded, for the reason given in Section 3.5. And the
detector and segmenter are excluded from the recognition cost and reported as a separate shared
line, because they are identical for every cell and would otherwise be attributed to whichever
backbone was being measured.

<!-- estimate_flops() returns MACs; its docstring states "Rough MAC count" (src/models.py:206).
     This manuscript reports MACs and never labels them FLOPs. -->

【待补: 推理延迟——单张图像、批大小 1、指定硬件、热身若干次后取中位数的实测值；
      须写明硬件型号与热身/计时次数】

## 4. Results

### 4.1 Retrieval performance of the six cells

Table 3 reports rank-1 and rank-5 for every combination of crop policy and backbone. The cells in
a row share a backbone and differ only in the crop; the cells in a column share a crop and differ
only in the network.

| Backbone | Crop | Rank-1 | Rank-5 | $\Delta_{\mathrm{SAM}}$ (rank-1) |
| --- | --- | ---: | ---: | ---: |
| Convolutional trunk | $I_{\mathrm{sam}}$ | 【待补】 | 【待补】 | — |
| Convolutional trunk | $I_{\mathrm{box}}$ | 【待补】 | 【待补】 | 【待补】 |
| Vision Transformer | $I_{\mathrm{sam}}$ | 【待补】 | 【待补】 | — |
| Vision Transformer | $I_{\mathrm{box}}$ | 【待补】 | 【待补】 | 【待补】 |
| ResNet-50 | $I_{\mathrm{sam}}$ | 【待补】 | 【待补】 | — |
| ResNet-50 | $I_{\mathrm{box}}$ | 【待补】 | 【待补】 | 【待补】 |

*Table 3. Retrieval accuracy on the frozen identity-disjoint split, 96 identities.
$I_{\mathrm{sam}}$ = mask-purified dorsal coat; $I_{\mathrm{box}}$ = detector box crop.
$\Delta_{\mathrm{SAM}}$ is rank-1 on the purified arm minus rank-1 on the box arm, computed per
backbone. Each figure is to be read from `outputs/<cell>/metrics.json` of the corresponding
completed run and cross-checked against `outputs/summary_six_cells.json`.*

<!-- PROVENANCE. No value in Table 3 exists yet. The placeholder tables in results/metrics.json
     and results/accuracy_curves.json are hard-coded constants generated by src/metrics.py and
     src/results.py, whose own docstrings call them placeholders, and whose disclaimer text was
     removed in commits 2e1e2d4/cdde8cc/ddaf922. They must not be used. See
     paper/evidence-audit.md §1 for the full account, including the fact that
     src/results.py:20-38 names the dictionary "_FINAL" while describing its contents as a
     narrative "scenario". -->

【待补: 每个单元格的 gallery/query 图像数与被评身份数（须与 Table 2 一致）】

【待补: 重复实验统计——至少 3 个随机种子的均值与标准差；若仅单次运行，须明确说明，
      并说明为何单次运行足以支撑结论】

### 4.2 Effect of mask-based purification

The per-backbone contrast $\Delta_{\mathrm{SAM}}$ is the primary outcome of the crop factor.
Because both arms share their detections, their identity assignment and their image count, the
comparison is paired at the level of the animal, and the appropriate summary is the difference
per backbone rather than a pooled difference.

【待补: 三个骨干各自的 Δ_SAM 数值，以及与之配套的自助法置信区间或配对检验结果】

【待补: 掩膜质量对 Δ_SAM 的影响——掩膜 IoU 或前景面积比例的分布，以及低质量掩膜样本上的
      分项结果。若未测量掩膜质量，须明确说明这是本研究的局限】

We note a design decision with a consequence for interpretation. The two arms share the identity
split, so an animal that is mis-detected in one arm is mis-detected in both, and the paired
difference is not inflated by easy animals appearing only in one arm. The cost of that decision is
that the design cannot detect an interaction between crop policy and detection quality; that would
require a third factor.

### 4.3 Computational cost

Table 4 gives the measured cost of each backbone at the common input resolution. These figures are
properties of the committed architecture definitions and were reproduced by instantiating each
model from the reference implementation and calling its own counting routines, so they do not
depend on any training run.

| Backbone | Parameters | Multiply–accumulates @ 224² | Relative MACs |
| --- | ---: | ---: | ---: |
| Convolutional trunk | 1,832,224 | 242,311,168 | 1.00× |
| Vision Transformer (ViT-Tiny/16) | 5,623,232 | 1,074,757,632 | 4.44× |
| ResNet-50 | 24,557,120 | 4,088,184,832 | 16.87× |

*Table 4. Cost of the inference embedding network at 224 × 224, embedding dimension 512,
classification head excluded. Parameters and MACs measured by `count_parameters` and
`estimate_flops` in `src/models.py` with `configs/experiment.yaml`; the relative column is the
ratio to the convolutional trunk. These are MACs, not FLOPs.*

<!-- Traceability for Table 4 — all three rows reproduced during the audit:
       yolo_arcface   params 1,832,224   MACs 242,311,168
       pruned_vit_arcface params 5,623,232 MACs 1,074,757,632
       wang_metric (resnet50) params 24,557,120 MACs 4,088,184,832
     Command: build_model(name, load_yaml("configs/experiment.yaml")) then
     count_parameters(model) and estimate_flops(model, 224, cpu).
     Relative ratios computed from these MACs: 1074757632/242311168 = 4.44,
     4088184832/242311168 = 16.87. See paper/evidence-audit.md §7. -->

The spread is the point. The three backbones differ by a factor of 13.4 in parameters and a factor
of 16.9 in multiply–accumulates, so an accuracy difference of a few points between them has to be
read against a very large difference in the cost of running them on an aircraft. For a UAV
payload with a power and mass budget, the convolutional trunk at 242 M MACs per frame is a
qualitatively different proposition from ResNet-50 at 4088 M MACs per frame, and the identification
accuracy that the larger model buys is the quantity that decides whether the trade is worth making.

<!-- The pruning stage described in the project's method draft (paper/method.tex:290-324) is NOT
     reflected in Table 4. In the committed revision the pruning operator removes nothing at the
     configured ratio: int(3 heads * 0.25) = 0 skips head pruning, and the MLP write uses
     advanced indexing, which returns a copy, so `w[idx].zero_()` leaves the parameter unchanged
     (0 of 192 intended rows zeroed; verified empirically). Pruned and unpruned parameter counts
     are identical at 5,623,232. The ViT row is therefore reported as an unpruned ViT-Tiny/16.
     See paper/evidence-audit.md §4. -->

【待补: 若在修订版中真正实现了结构化剪枝，须补充剪枝率—参数量—MACs—Rank-1 的对应表，
      并说明剪枝后是否重新微调及微调设置】

### 4.4 Training behaviour

Rank-1 and rank-5 are logged every five epochs, which gives a coarse view of whether a cell has
converged within the 30-epoch budget or is still improving when training stops.

【待补: 六个单元格的 Rank-1/Rank-5 随轮次变化曲线（数据来源 outputs/<cell>/rank_callback.json），
      以及是否出现未收敛的单元格】

<!-- Do not use results/accuracy_curves.json. Those curves are placeholders drawn at epochs
     200/400/600/800/1000, a schedule that contradicts the 30-epoch budget in
     configs/experiment.yaml (paper/evidence-audit.md §8). -->

A convergence diagnostic matters for the interpretation of Section 4.2: if one backbone is still
improving at epoch 30 while another has plateaued, the between-backbone comparison at a fixed
epoch budget understates the former, and the honest reading is that the budget, not the
architecture, is binding. The same caveat applies to the crop contrast, since the purified arm has
fewer informative pixels and may converge on a different trajectory.

## 5. Discussion

### 5.1 Why the crop question is an agronomic question

It is tempting to treat the choice between a masked crop and a box crop as an implementation
detail of a computer-vision pipeline. In a pasture setting it is better understood as a question
about what the system is allowed to learn, and the answer has direct management consequences.

Cattle in a commercial herd are not randomly distributed in space. They are grouped by age, by
sex, by weight band and by treatment status, and those groupings are physically realised as pens,
paddocks and feeding stations. A network trained on box crops drawn from such a layout has access
to the pen as a feature, and the pen is correlated with identity precisely because management put
the animals there. On the farm where the model was trained, this correlation makes the model look
better than it is. On the next farm — a different layout, a different grouping, a different set of
animals — the correlation is absent or reversed, and the model degrades in a way that is hard to
diagnose because nothing in the training pipeline flagged the dependence.

Mask-based purification is the most direct intervention against this failure. By restricting the
input to the dorsal coat, it removes the pen, the fence and the ground from the model's view, and
it forces the representation to be built from the animal. If $\Delta_{\mathrm{SAM}}$ is positive,
that is evidence that the shortcut was being exploited and that removing it helped; if it is
negative or near zero, that is evidence that the context was carrying information the coat alone
does not, or that the mask is discarding usable coat pixels at oblique view angles. Either outcome
is informative for a farm, because either outcome tells the operator whether the system's accuracy
will survive a change of premises.

There is a second, more mundane reason to care. A box crop that includes a distinctive
ground feature is easy to obtain — it requires only a detector — whereas a masked crop requires a
segmentation stage that must run on the aircraft or on an edge device, at a cost in power and
latency that competes with the recognition network itself. Whether that cost is justified is
exactly the question $\Delta_{\mathrm{SAM}}$ answers, and answering it per backbone is necessary
because a large network may be able to learn to ignore the background that a small one cannot.

### 5.2 What the cost figures imply for deployment

The three backbones in Table 4 span a factor of nearly seventeen in multiply–accumulates. On a
UAV, that factor translates into flight endurance, into the thermal envelope of the payload
enclosure, and into whether inference can be done on board at frame rate or must be deferred to a
ground station after landing. A deferred pipeline is not necessarily worse — a paddock can be
flown in the morning and processed at midday, and for growth monitoring the latency is
irrelevant — but it changes the system architecture, because on-board inference permits
adaptive flight (re-flying an animal whose image was rejected) whereas deferred processing
does not.

The practical reading of Table 4 is therefore conditional on the intended workflow. For a
post-flight batch pipeline, the ResNet-50 row is affordable and the question is purely whether
its accuracy justifies its cost. For on-board inference at a few frames per second on an embedded
accelerator, the convolutional trunk is the only one of the three that fits comfortably, and the
relevant question becomes how much accuracy is lost by choosing it. This is why the protocol
reports accuracy and cost together rather than accuracy alone: the comparison that matters
operationally is between two cells, not between two numbers.

### 5.3 Identification as part of a monitoring record

An identification accuracy is not, by itself, a management capability. What a livestock operation
needs is a record: this animal, on this date, had this estimated liveweight, this activity level,
this locomotion score. The identification step is the join key that makes such a record possible,
and its error characteristics propagate into every downstream statistic.

Two properties of that propagation deserve emphasis. First, identification errors are not
independent of the measurements they accompany: an animal that is partly occluded, or imaged at an
oblique angle, or standing in shadow is both harder to identify and harder to measure, so
identification failures cluster on the same images where the phenotypic measurement is least
reliable. A pipeline that silently drops failed identifications will therefore produce a biased
subsample that looks cleaner than the herd. Recording the rejection rate, and reporting it
alongside accuracy, is a minimum safeguard.

Second, rank-$k$ retrieval and closed-set classification answer different operational questions.
If the system acts autonomously — dosing, drafting, or attributing a weight to an individual
without human review — then the relevant figure is rank-1 with an explicit rejection threshold,
and the cost of an error is asymmetric. If the system proposes a shortlist to a stockperson who
confirms it, then rank-5 and the length of the list are the relevant figures, and the human is
absorbing the residual error. Our protocol reports both because both correspond to real
deployments, but a reader should not interpret a rank-5 figure as if it were a rank-1 figure.

### 5.4 Limitations and threats to validity

Several limitations bound what can be concluded, and they are stated here rather than left for a
reader to infer.

*Scale and breed.* The study uses 96 Simmental beef cattle from a single archive. Simmental coats
are relatively high-contrast and patterned, which is favourable for coat-based identification;
darker, more uniform-coated breeds, and coat patterns that change substantially between seasons,
present a harder problem that this dataset cannot address. Ninety-six identities is a small
enrolment by the standards of person re-identification, and rank-1 on a 96-way problem is not
directly comparable to rank-1 on a 1000-way problem.

*Single site.* All imagery comes from one farm. The transfer question raised in Section 5.1 — does
the model survive a change of premises — is not answered by a single-site experiment, and the
$\Delta_{\mathrm{SAM}}$ result should be read as evidence about mechanism rather than as a
guarantee of transfer.

*Model selection on the evaluation split.* The retained checkpoint is the one with the best
observed rank-1 on the gallery/query splits, as described in Section 3.6. The reported figures are
therefore mildly optimistic in absolute terms. The bias is applied identically to all six cells and
so should not affect the ordering, but a fully clean protocol would carve a validation split for
this purpose. The reference implementation defines no validation split, which is a shortcoming of
the protocol as implemented.

*Epoch budget.* A fixed 30-epoch budget with no schedule was chosen to keep the comparison
symmetric. If some cells have not converged, the budget rather than the architecture is binding,
and the between-backbone comparison is correspondingly weakened. Reporting the training curves
(Section 4.4) is the mitigation, not a solution.

*Segmentation dependence.* The purified arm inherits the failure modes of the segmenter. A mask
that clips the shoulder, or that leaks into an adjacent animal, changes what the recogniser sees in
a way that is correlated with pose and crowding. The protocol does not currently measure mask
quality, and doing so is the most obvious single improvement to it.

*No open-set evaluation.* All probes are enrolled animals. A system deployed on a farm will
encounter animals that are not in its gallery — newly purchased, visiting, or a neighbouring herd
within the flight line — and the closed-set metrics reported here do not characterise behaviour in
that case. Open-set evaluation with an explicit rejection threshold is required before autonomous
use.

*Statistical power.* The number of independent experimental units is small: six cells, and within
each cell a few hundred query images drawn from 96 animals. Query images of the same animal are not
independent, so confidence intervals computed over images understate the uncertainty. Animal-level
resampling is the appropriate basis for inference and should be used when the results are reported.

### 5.5 What would change the conclusions

For transparency we state in advance which observations would overturn the interpretation we have
set out, so that the results section is read as a test rather than as a confirmation.

If $\Delta_{\mathrm{SAM}}$ is strongly positive for the small backbone and near zero for the large
one, the reading is that the shortcut is learnable but only by a network with enough capacity to
find it, and that a small model is incidentally robust — a conclusion with a clear practical
consequence, namely that model capacity and crop policy are not independent design choices. If
$\Delta_{\mathrm{SAM}}$ is negative for all three backbones, the reading is that mask-based
purification is not a worthwhile preprocessing cost for this biometric, and the box crop should be
preferred on efficiency grounds. If the ordering of backbones inverts between the two arms, then
neither the crop nor the backbone can be selected independently, and any published single-arm
comparison should be treated as confounded. Each of these outcomes is consistent with the design,
which is the property a controlled experiment is supposed to have.

## 6. Conclusions

Individual identification is the join key that turns aerial imagery of a paddock into per-animal
management records, and the coat pattern visible from above is a practical biometric for the
purpose. The obstacle to progress is not the absence of recognition architectures but the absence
of comparability: reported accuracies conflate how much of the animal the network sees with how
much capacity it has.

We have specified and implemented a 2 × 3 protocol that separates those two factors. Two paired
crops are derived from the same detections in the same frames, so the arms differ in retained
pixels and nothing else; three architecturally distinct backbones are trained under one additive
angular margin objective with shared scale, margin, embedding dimension, optimiser and epoch
budget; and one frozen identity-disjoint split serves every cell. Cost is reported on the
inference network alone, excluding the training classifier, in multiply–accumulates rather than
FLOPs. Under that accounting the three backbones cost 1.83 M, 5.62 M and 24.56 M parameters at
242 M, 1075 M and 4088 M MACs per 224 × 224 frame.

The contribution is a measurement instrument rather than a winning number. Applied to aerial
cattle imagery it yields a per-backbone estimate of whether mask-based purification of the dorsal
coat helps, which is a question about whether the recogniser is reading the animal or the pen; and
it yields an accuracy-versus-cost curve that an operator can use to decide between post-flight
batch processing and on-board inference. The retrieval figures for the six cells complete the
instrument, and the interpretation we expect them to support is set out in Section 5.5 in a form
that the measurements can contradict.

Three things would make a follow-on study substantially stronger: replication across farms and
breeds, so that transfer rather than within-site accuracy becomes the reported quantity;
open-set evaluation with an explicit rejection threshold, so that unknown animals are handled
rather than assumed away; and a measurement of segmentation quality, so that the purified arm's
dependence on its mask is characterised rather than ignored. None of these requires a new
architecture. All three would make the resulting numbers usable as evidence in a livestock
management decision, which is the standard that an applied agricultural journal should hold them
to.

## CRediT authorship contribution statement

**Tianyi Ma:** Conceptualization, Methodology, Software, Validation, Formal analysis,
Investigation, Data curation, Writing – original draft, Writing – review & editing, Visualization,
Project administration, Funding acquisition.

<!-- Single-author manuscript per CITATION.cff and README.md. Role assignment above follows the
     CRediT taxonomy and should be confirmed by the author before submission; if additional
     contributors are added their roles must be listed separately.
     【待补: 作者确认 CRediT 角色分配；如有其他作者须逐一补充】 -->

## Declaration of competing interest

The authors declare that they have no known competing financial interests or personal
relationships that could have appeared to influence the work reported in this paper.

## Acknowledgements

This work was supported by the Songshan Laboratory (No. 241110210200) and the Major Science and
Technology Special Project of Henan Province (No. 221100210600).

<!-- Funding numbers as stated in README.md:340-342. -->

The UAV cattle archive used in this study was deposited by Yaowu Wang and colleagues, and the
authors gratefully acknowledge their decision to release both the imagery and the accompanying
identification code and logs under open licences.

## Data availability

The imagery used in this study is publicly available from Zenodo under CC BY 4.0:
Wang, Y., 2024. Dataset for cattle growth monitoring in PLF using UAV-based LiDAR.
https://doi.org/10.5281/zenodo.13364663. The accompanying code, results and logs for the
coat-pattern identification work are available as https://doi.org/10.5281/zenodo.13361729.

The reference implementation of the protocol described in this paper, including the paired-crop
manifest construction, the three backbones, the shared ArcFace objective, the retrieval metrics and
the cost accounting, is distributed under the Apache License 2.0 at
https://github.com/tianyi-star/2d_cattle.

<!-- The implementation is released, but no trained checkpoint or measured result file is
     distributed with it as of the audited revision (outputs/ and data/ are gitignored and were
     never committed). A data availability statement that promised measured artefacts would be
     inaccurate; the statement above promises code only. -->

【待补: 若期刊要求，提供训练好的模型权重与完整结果文件的存放地址（Zenodo/Figshare），
      并在投稿前确认仓库中 results/ 目录下的占位文件已被删除或明确标注，避免读者误读】

---

## References

Andrew, W., Greatwood, C., Burghardt, T., 2017. Visual localisation and individual identification
of Holstein Friesian cattle via deep learning, in: Proceedings of the IEEE International Conference
on Computer Vision Workshops (ICCVW), pp. 2850–2859.
https://doi.org/10.1109/ICCVW.2017.336

Andrew, W., Gao, J., Mullan, S., Campbell, N., Dowsey, A.W., Burghardt, T., 2021. Visual
identification of individual Holstein-Friesian cattle via deep metric learning. Computers and
Electronics in Agriculture 185, 106133. https://doi.org/10.1016/j.compag.2021.106133

Bezen, R., Edan, Y., Halachmi, I., 2020. Computer vision system for measuring individual cow feed
intake using RGB-D camera and deep learning algorithms. Computers and Electronics in Agriculture
172, 105345. https://doi.org/10.1016/j.compag.2020.105345

Chen, T., Liu, M., Si, Y., Khot, L.R., Wang, K., Wang, B., 2026. Multimodal deep learning framework
for robust individual identification of Holstein cattle using gait dynamics and coat pattern
features. Computers and Electronics in Agriculture 246, 111632.
https://doi.org/10.1016/j.compag.2026.111632

Deng, J., Guo, J., Xue, N., Zafeiriou, S., 2019. ArcFace: Additive angular margin loss for deep face
recognition, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition
(CVPR), pp. 4685–4694. https://doi.org/10.1109/CVPR.2019.00482

Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani,
M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., Houlsby, N., 2021. An image is worth 16x16
words: Transformers for image recognition at scale, in: International Conference on Learning
Representations (ICLR). arXiv:2010.11929. 【需核对: ICLR 会议论文集无 DOI，页码需按正式版本核对】

Han, S., Fuentes, A., Yoon, S., Jeong, Y., Kim, H., Park, D.S., 2023. Deep learning-based
multi-cattle tracking in crowded livestock farming using video. Computers and Electronics in
Agriculture 212, 108044. https://doi.org/10.1016/j.compag.2023.108044

He, K., Zhang, X., Ren, S., Sun, J., 2016. Deep residual learning for image recognition, in:
Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 770–778.
https://doi.org/10.1109/CVPR.2016.90

He, S., Luo, H., Wang, P., Wang, F., Li, H., 2021. TransReID: Transformer-based object
re-identification, in: Proceedings of the IEEE/CVF International Conference on Computer Vision
(ICCV), pp. 14993–15002. https://doi.org/10.1109/ICCV48922.2021.01474

Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S.,
Berg, A.C., Lo, W.-Y., Dollár, P., Girshick, R., 2023. Segment Anything, in: Proceedings of the
IEEE/CVF International Conference on Computer Vision (ICCV), pp. 3992–4003.
https://doi.org/10.1109/ICCV51070.2023.00371

Kuncheva, L.I., 2021. Animal reidentification using restricted set classification. Ecological
Informatics 62, 101225. https://doi.org/10.1016/j.ecoinf.2021.101225

Li, G., Erickson, G.E., Xiong, Y., 2022. Individual beef cattle identification using muzzle images
and deep learning techniques. Animals 12, 1453. https://doi.org/10.3390/ani12111453

Li, Z., Lei, X., Liu, S., 2022. A lightweight deep learning model for cattle face recognition.
Computers and Electronics in Agriculture 195, 106848.
https://doi.org/10.1016/j.compag.2022.106848

Liu, Y., Li, W., Liu, X., Li, Z., Yue, J., 2024. Deep learning in multiple animal tracking: A
survey. Computers and Electronics in Agriculture 224, 109161.
https://doi.org/10.1016/j.compag.2024.109161

Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., Guo, B., 2021. Swin Transformer:
Hierarchical vision transformer using shifted windows, in: Proceedings of the IEEE/CVF
International Conference on Computer Vision (ICCV), pp. 9992–10002.
https://doi.org/10.1109/ICCV48922.2021.00986

Pathak, P.D., Prakash, S., 2025. Attention-based multi-modal robust cattle identification technique
using deep learning. Computers and Electronics in Agriculture 238, 110747.
https://doi.org/10.1016/j.compag.2025.110747

Rahman, Z.U., Asaari, M.S.M., Ibrahim, H., 2025. Advancing animal farming with deep learning: A
systematic review. Computers and Electronics in Agriculture 237, 110674.
https://doi.org/10.1016/j.compag.2025.110674

Redmon, J., Divvala, S., Girshick, R., Farhadi, A., 2016. You Only Look Once: Unified, real-time
object detection, in: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition
(CVPR), pp. 779–788. https://doi.org/10.1109/CVPR.2016.91

Rohan, A., Rafaq, M.S., Hasan, M.J., Asghar, F., Bashir, A.K., Dottorini, T., 2024. Application of
deep learning for livestock behaviour recognition: A systematic literature review. Computers and
Electronics in Agriculture 224, 109115. https://doi.org/10.1016/j.compag.2024.109115

Shumaly, M., Park, Y., Agha, S., Pandey, S., Steibel, J., 2025. Computer vision-based animal
phenotyping and analysis in presence of uncertain identification. Computers and Electronics in
Agriculture 237, 110560. https://doi.org/10.1016/j.compag.2025.110560

Tian, M., Guo, H., Chen, H., Wang, Q., Long, C., Ma, Y., 2019. Automated pig counting using deep
learning. Computers and Electronics in Agriculture 163, 104840.
https://doi.org/10.1016/j.compag.2019.05.049

Wang, C.-Y., Yeh, I.-H., Liao, H.-Y.M., 2024. YOLOv9: Learning what you want to learn using
programmable gradient information, in: Computer Vision – ECCV 2024, Lecture Notes in Computer
Science, pp. 1–21. https://doi.org/10.1007/978-3-031-72751-1_1

Wang, Y., Mücher, S., Wang, W., Guo, L., Kooistra, L., 2023. A review of three-dimensional computer
vision used in precision livestock farming for cattle growth management. Computers and Electronics
in Agriculture 206, 107687. https://doi.org/10.1016/j.compag.2023.107687

Wang, Y., 2024. Dataset for cattle growth monitoring in PLF using UAV-based LiDAR. Zenodo.
https://doi.org/10.5281/zenodo.13364663

Wang, Y., Mücher, S., Wang, K., Wang, W., Kooistra, L., 2025. Deep metric learning for individual
cattle identification using coat patterns: Proposal for a best practice. Computers and Electronics
in Agriculture 238, 110754. https://doi.org/10.1016/j.compag.2025.110754

Yan, J., Tetila, E.C., Zhao, L., Gonçalves, R.C., Castanheiro, L.F., Valem, L.P., Barbedo, J.G.A.,
2026. Deep learning-based UAV framework for automated morphological and growth analysis of feedlot
cattle. Computers and Electronics in Agriculture 245, 111559.
https://doi.org/10.1016/j.compag.2026.111559

Zhao, B., Xiao, B., Liu, C., Jin, X., Ye, J., Yu, M., 2026. Bovine-LTID: A vision-based
spatio-temporal identity system for multi-camera cattle tracking and long-term behavior monitoring.
Computers and Electronics in Agriculture 253, 112051.
https://doi.org/10.1016/j.compag.2026.112051

<!-- =================================================================== -->
<!-- Reference verification. Every entry above was resolved through the    -->
<!-- Crossref REST API during preparation (DOIs, volumes, pages and       -->
<!-- author lists taken from the returned records, not from memory).      -->
<!-- Journal-hosted entries: 16 of 27 are Computers and Electronics in     -->
<!-- Agriculture papers. The two entries carrying 【需核对】 are flagged    -->
<!-- because no authoritative Crossref record exists for the exact venue:  -->
<!-- the ViT paper (ICLR proceedings, no DOI) and, if retained, the        -->
<!-- MegaDescriptor preprint (arXiv only).                                 -->
<!-- See paper/evidence-audit.md §12 for the DOI list.                     -->
<!-- =================================================================== -->
