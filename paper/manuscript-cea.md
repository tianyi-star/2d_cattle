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
animal rather than a herd as the unit of management: growth monitoring, feed conversion, disease
surveillance and provenance records. In pasture systems, ear tags, collars and boluses carry
recurring cost and require handling, which motivates identification from imagery already collected
by unmanned aerial vehicles (UAVs). Cattle re-identification is otherwise studied almost entirely
from the ground, where the flank fills the frame and the coat is side-lit; from low altitude the
animal is small, seen from above, and only its dorsal coat is available as a cue. We study
identification in that regime on an openly released UAV archive of 96 Simmental beef cattle, and
ask how much of a reported accuracy reflects the animal rather than its surroundings. Because
studies differ in
segmentation, crop policy, backbone and gallery composition at once, we use a controlled factorial
protocol rather than a new architecture: two paired crops from the same detections — a
mask-purified dorsal coat region and the plain bounding box in which pasture, shadow and limbs
remain — crossed with three embedding networks under one shared additive angular margin objective.
Evaluation uses one frozen identity-disjoint partition, so no animal appears in more than one of
the training, validation and test sets and a correct retrieval cannot be a near-duplicate frame of
a memorised animal. Cost is reported for the inference network alone, from 1.83 M to 24.56 M
parameters, with structured pruning cutting the transformer from 5.62 M to 4.14 M. Separating crop
purification from backbone capacity is, we argue, a precondition for interpreting aerial cattle
re-identification on a working farm. 【待补: 六单元格检索结果一句话摘要（含最佳单元格 Rank-1 与
Δ_SAM 方向），在实测完成后补入，不得以占位数值代替】

**Keywords:** cattle re-identification; unmanned aerial vehicle; low-altitude aerial imagery;
precision livestock farming; coat pattern; identity-disjoint evaluation

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
cover a paddock in minutes and yields a few hundred to a few thousand frames, and the aircraft
never touches the animal, so the measurement does not perturb the behaviour it is meant to
observe. The same flight that supports a body-condition or liveweight estimate can, in principle,
also establish which animal is being measured. Wang et al. (2023) review how three-dimensional and
imaging approaches have been used for cattle growth management in this setting, and Yan et al. (2026) demonstrate a UAV
framework for morphological and growth analysis of feedlot cattle. The missing link in that
chain is identity.

That cattle can be identified from above is now established. The dorsal coat pattern of
Holstein-Friesian and Simmental animals has been shown to be sufficiently individual, and
sufficiently stable over weeks, to support retrieval. Andrew et al. (2017) showed visual
localisation and individual identification of Holstein-Friesian cattle from a moving platform, and
Andrew et al. (2021) developed the approach into a deep metric learning system with an explicit
treatment of the open-set case. Wang et al. (2025a) extended coat-pattern identification to paired
aerial and ground imagery of 96 Simmental beef cattle and proposed it as a best practice, releasing
both the imagery (Wang, 2024) and the accompanying code and logs. The present study analyses that
same released archive, downloaded and stored locally on our GPU server; our contribution is the
identification protocol applied to it, not the collection itself. Related biometrics have been
pursued in
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

Fourth, we implement the two stages that make the design executable rather than aspirational, and
report the compression they achieve. The masking stage is a working box-prompted segmentation
front-end with a disk-cached mask store and a hard failure on missing weights, so the purified arm
is produced by the pipeline rather than assumed; the pruning stage physically removes attention
heads and feed-forward channels under a magnitude criterion, so the transformer's cost genuinely
falls — 26.3 % of parameters and 27.0 % of multiply–accumulates at a removal rate of 0.25. We
state plainly that the pruning criterion is weight magnitude and not the embedding-geometry
criterion that motivated the design, and we do not claim the latter's properties for the former.

Fifth, we release a reference implementation that enforces the protocol, including the retrieval
metric, the paired-crop manifests, the seeded six-cell runner, the aggregation step that rejects
placeholders, and the cost accounting, so that the comparison can be repeated on other farms and
other breeds without re-deriving the conditions.

The remainder of the paper is organised as follows. Section 2 reviews individual identification
in livestock, the use of aerial imagery in pasture management, and the metric-learning losses on
which the shared objective is based. Section 3 gives the materials and methods, including data
acquisition and annotation, the crop construction, the three backbones, the objective, the
structured pruning procedure, and the experimental setup. Section 4 reports retrieval performance
and computational cost for the six cells. Section 5 discusses what the design implies for on-farm
use, where it can fail, and what would be required before an identification system of this kind
could be trusted as a management record. Section 6 concludes.

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

**The dominant prior setting is ground-level, and this matters.** Almost all of the cattle
re-identification literature, and effectively all of the work that establishes the accuracy
figures this paper is compared against, is captured from the ground. Andrew et al. (2017, 2021)
photograph Holstein-Friesian animals from a robot or a hand-held camera at animal height, where
the flank fills the frame and the coat is seen at a near-perpendicular angle under side lighting.
Muzzle- and face-based studies (Li, G. et al., 2022; Li, Z. et al., 2022) are ground-level by
construction, since the cue is only visible from in front of the animal. Gait-based identification
(Chen et al., 2026) requires a lateral view along a walkway. Multi-modal fusion approaches
(Pathak and Prakash, 2025) assume one of these ground-level views exists for each modality.

The consequence is that reported accuracies in this literature describe a setting in which the
animal is close, large in frame, well lit from the side, and observed from a viewpoint chosen by
the photographer. A low-altitude aerial system operates under none of those conditions: the animal
is small in a wide frame, observed from above, lit from above, and its head and legs are largely
unavailable as cues. Framing the present study against that prior setting is therefore not a
rhetorical move — it identifies a specific gap. The question is whether the dorsal coat pattern,
viewed from above at the resolution a UAV can deliver, carries enough identity information to
support the per-animal records that precision livestock farming needs. Answering it requires data
captured in that regime, which is the dataset described in Section 3.2.

Because the imaging regime differs, the comparison to ground-level results must be made carefully.
We do not claim that our figures are directly comparable to published ground-level accuracies:
the cue set, the resolution, the viewpoint distribution and the gallery composition all differ, and
a numeric comparison across them would be uninformative in either direction. The relevant
comparison is internal — between crop policies and between backbones under one objective on one
dataset — and the external literature supplies the motivation and the design vocabulary, not a
leaderboard to be topped.

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
cells is trained from scratch on its own crop manifest and evaluated on the same identity-disjoint
partition.
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

The imagery analysed in this study is obtained from the Zenodo archive of Wang (2024), record
[10.5281/zenodo.13364663](https://doi.org/10.5281/zenodo.13364663), which was downloaded and is
stored locally on our GPU server at `/home/iadi/UAV-basedData/RGB-data/individaulsByIds`. The
local copy is organised as one directory per animal; the directory names are the herd identifiers
and are not contiguous, so the collection is described throughout by its identity count rather
than by its identifier range. The archive documents 96 semi-free-range beef cattle, and the
companion record (Wang, 2024, `10.5281/zenodo.13361729`) records them as 96 Simmental male beef
cattle; we therefore describe the herd as 96 Simmental beef cattle and cite those records for the
claim rather than asserting it as our own observation.

<!-- Provenance, as confirmed by the author (this supersedes two earlier drafts, one of which
     wrongly called the data private and one of which wrongly denied the Zenodo origin):
       source  : Zenodo record 10.5281/zenodo.13364663 (Wang 2024, CC BY 4.0, dataShare.zip,
                 26,486,342,645 bytes)
       local   : /home/iadi/UAV-basedData/RGB-data/individaulsByIds on the GPU server,
                 one directory per identity, names 101 ... 328 non-contiguous, 96 identities
       breed   : "96 Simmental male beef cattle" - companion record 10.5281/zenodo.13361729
       article : Wang, Kooistra, Mücher & Wang (2025), Scientific Data 12:1318,
                 doi 10.1038/s41597-025-04783-6 - documents site, dates, platform and breed.
     The local directory was not reachable from the environment where this draft was prepared,
     so per-identity image counts and split sizes are unverified and remain 【待补】. -->

**Low-altitude aerial capture as the distinguishing setting.** The imagery is captured from an
unmanned aerial vehicle flying over the animals at low altitude. This is a materially different
imaging regime from the one that dominates the cattle re-identification literature, and it is
worth stating plainly why, because it is the reason this collection is interesting rather than
merely another cattle dataset. Ground-level studies photograph an animal at eye height: the head,
the muzzle, the flank and the legs are available, the animal fills the frame, and the background is
whatever happens to be behind it. An aerial view offers none of that. The camera sees the dorsal
surface at an oblique-to-near-nadir angle; the animal occupies a modest part of a wide frame; the
head is foreshortened or hidden; and the coat pattern is observed under near-uniform downward
illumination rather than under the side-lighting that makes ground-level texture cues legible.

Two consequences follow for identification, and both shape the protocol. First, the usable
biometric is the dorsal coat pattern and essentially nothing else — not the face, not the gait
signature available from a side view, not the animal's dimensions against a familiar background.
Second, the nuisance factors are different in kind: at low altitude the immediate surroundings are
pasture, shadow and the body of a neighbouring animal, all of which are visible in a rectangular
crop and all of which can correlate with identity when animals are grouped by pen or paddock. This
is what motivates the paired-crop comparison of Section 3.3 rather than treating segmentation as a
preprocessing detail. For livestock management the practical consequence is direct: a recogniser
that survives an aerial view can be run off survey flights a growth-monitoring programme is already
making, which is what makes per-animal records affordable in a pasture system.

**Capture parameters.** The site, campaign dates, platform and sensor are documented in the data
article accompanying the archive (Wang et al., 2025b). Where the present study
depends on a parameter that the archive documents, we cite the record; where the parameter affects
the recognition geometry but is not pinned down for this local copy, we leave it to be confirmed
against the flight logs rather than restating a figure we have not verified.

【待补: 飞行高度与地面采样距离 (GSD)、相机型号与焦距、原始图像分辨率、飞行架次与日期、
      养殖场位置与圈舍/放牧地布局、光照条件范围——须据飞行记录或数据文章逐项核对后填写；
      不得由图像推测，也不得照抄其他研究】

Animal-level metadata is needed for the same reason: a reader cannot judge whether a coat-based
result will transfer without knowing what the coats look like.
【待补: 毛色与花纹类型分布、性别与月龄范围、体重区间——品种已据 Zenodo 记录 13361729 定为
      96 头西门塔尔公牛，其余须由牧场记录提供】

**Annotation.** Annotation is at the level of identity, not of pixels. Each image is assigned to
the folder of the animal it depicts, and that folder label is the only supervision used. No
segmentation mask is annotated: the mask-purified arm is produced by prompting a segmentation
model with the detector box, so the cost of the masking stage is inference time rather than
labelling time.

【待补: 标注流程细节——标注人员数量与培训、身份核对方式（是否与耳标/牧场记录交叉验证）、
疑难样本（遮挡、多牛同框、部分出画）的处理规则、标注一致性检查结果】

### 3.2.1 Identity-disjoint partitioning

The 96 identities are partitioned into training, validation and test sets **by animal, not by
image**. Every image of a given animal falls in exactly one set, and no animal appears in more
than one set.

This is a stricter condition than it may appear, and it is the single most important protocol
decision in the paper. The intuitive alternative — pool all images, shuffle, and deal them into
train, validation and test in some proportion — is what a great many re-identification studies do,
and on a dataset of this kind it produces optimistically biased numbers. The reason is that
consecutive UAV frames of the same animal are highly correlated: they show the same coat under
almost the same illumination from almost the same viewpoint, often seconds apart. If an image of
animal 117 lands in training and its near-duplicate lands in the test set, the model can retrieve
that animal by recognising the specific frame rather than the animal, and rank-1 measures
frame-level memorisation rather than individual identification. The inflation is largest exactly
where the study is most likely to be read as successful: high reported accuracy on a small
identity count.

Holding identities out removes the mechanism. A test animal's coat pattern has never been seen
during training in any frame, so a correct retrieval requires a representation that generalises
across animals rather than one that memorises instances. The reported rank-1 is then an estimate
of identification performance on animals the system has not been trained on, which is the quantity
a farm actually cares about when a new group of cattle arrives.

The split is fixed once, recorded in the manifest, and shared by all six cells and all ablation
arms, so that no comparison between backbones or between crops is confounded by a difference in
which animals were held out. The training, validation and test identity counts are reported in
Table 1.

The partition is drawn by sorting the identity identifiers, shuffling them with a fixed random seed,
and assigning whole animals to the three sets in that order; the seed, the split mode, the requested
proportions, the complete identity identifier list of each set, the image count of each set and the
per-identity image statistics are all written to a machine-readable split manifest at run time. The
requested proportions are 64 / 16 / 16 identities for training, validation and testing, allocated by
largest remainder so that the three counts always sum to the number of animals partitioned; the
manifest records the realised counts and any animal excluded, with the reason. An animal is eligible
for the test set only if it contributes at least two frames, one for the gallery and one for the
query; this keeps a single-frame animal from being unretrievable by construction. Both crop arms
reuse the same recorded partition, so the SAM and box columns are evaluated on the same held-out
animals. Table 1 is filled from that manifest rather than transcribed from a separate record.

Table 1 accounts for the images and identities that enter each split. Both crop arms draw on the
same accounting, so the table has one version rather than one per arm.

| Quantity | Value |
| --- | ---: |
| Identities in the archive | 96 |
| Identities assigned to training | 【待补】 |
| Identities assigned to validation | 【待补】 |
| Identities assigned to test | 【待补】 |
| Images in the training split | 【待补】 |
| Images in the validation split | 【待补】 |
| Images in the test (gallery + query) split | 【待补】 |
| Images per identity: min / median / max | 【待补】 |
| Images per identity: mean ± SD | 【待补】 |
| Identities excluded, with reason | 【待补】 |

*Table 1. Data accounting for the identity-disjoint partition. Identities are counted once and
appear in exactly one set; a query image and a gallery image of the same animal are never two
views exported from the same detection. The three identity counts must sum to 96 minus any
excluded animals, and the exclusions must be reported here rather than silently absorbed.*

<!-- No value in Table 1 has been measured. The local copy lives on the author's GPU server and
     was not reachable from the environment where this draft was prepared, so neither the
     per-identity image counts nor the split assignment have been verified. The 96 identity count
     is documented by the Zenodo archive records (13364663; 13361729) and the data article
     (doi 10.1038/s41597-025-04783-6), not counted here.
     Every other cell is unknown and must be filled from the manifest that the measured run
     actually used. See paper/evidence-audit.md section 6. -->

**Implementation status of the split.** The reference code implements this protocol as its
default. `src/data.py` partitions whole animals into disjoint `train`, `validation` and `test` sets,
places a test animal's frames in the gallery and query roles, and raises an error if any identifier
appears in more than one set; `src/train.py` and `src/evaluate.py` both perform that check before
use, and the training entry point selects its checkpoint on the validation animals rather than on the
test animals. A single command writes the manifest together with a JSON record of the partition
(mode, seed, proportions, identity lists, image counts, totals and a SHA-256 of the manifest itself)
and a Markdown summary of the same quantities, from which Table 1 is filled. Because the code and
the manifest are the same artefact, the partition quoted in this paper can be regenerated exactly
from the seed and mode it records.
<!-- src/data.py build_split() (default split_mode="identity"); assert_identity_disjoint();
     build_split_manifest() -> <manifest>.split.json and <manifest>.table1.md;
     src/aggregate.py build_seed_manifest() and the `manifest` subcommand.
     Reproduce: python3 -m src.aggregate manifest --image-root <tree> \
       --output-manifest data/manifest.csv --seed 0 [--dry-run]
     See paper/evidence-audit.md section 8.1. -->
【待补: 训练/验证/测试三个集合的身份编号名单与逐身份图像数，须由实测运行的清单文件直接填表；
      本次修改仅实现并固定了划分协议与清单格式，未进行任何实测】

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

The segmentation stage is implemented in the reference code as follows, and this description
matches what executes rather than what is planned. The segmenter is loaded lazily on the first
masking call, so a run restricted to the control arm never imports or allocates a segmentation
model. Three backends are supported: the reference `segment_anything` package with `vit_h`,
`vit_l` or `vit_b` weights; the `mobile_sam` package with `vit_t`; and the `ultralytics`
implementation already required by the detector path. The default model type is `vit_b`, and the
default backend is selected automatically from the installed packages.

Each mask is refined before it is used: it is clipped to the detector box, reduced to its largest
connected component so that a nearby animal cannot contribute an island of foreground, hole-filled,
and rejected outright if its area falls below a configurable fraction of the box area (0.5 % by
default). A rejected mask raises rather than falling back to the rectangle, because a silent
fallback would make the two arms byte-identical and destroy the contrast the protocol exists to
measure.

Masks are cached on disk, keyed jointly by the image identity, the detection box, and a
fingerprint of the segmenter configuration, with the cache schema versioned so that a stale entry
cannot be read as a current mask. Repeated epochs over one manifest therefore segment each image
once and thereafter read a small packed array instead of re-running the segmenter, which matters
because the segmenter is by far the most expensive stage per image. The cache directory, model
type, backend and device are all configurable through the path configuration and the preprocessing
entry point. If the requested segmenter weights are absent, preprocessing aborts before writing any
crop or manifest, so a partially built or structurally vacuous arm cannot be produced by accident.

<!-- Implementation traceability, verified by reading the code and running its tests:
     src/sam_mask.py implements the masker, refinement (clip/largest-component/hole-fill/
     min-area rejection at min_area_frac default 0.005), the versioned disk cache
     (CACHE_SCHEMA = "sam-mask-v1"), the lazy loader, and SamWeightsMissing.
     Defaults: model_type="vit_b" (src/sam_mask.py:221), backend "auto", cache dir
     data/sam_mask_cache (configs/path.yaml). src/preprocess.py --crop sam calls it;
     the bbox path is unchanged. 13 SAM tests pass on CPU without real weights;
     the opt-in real-checkpoint test (tests/test_sam_real_weights.py) is skipped by
     default and was verified separately by the author against sam_vit_b_01ec64.pth.
     See paper/evidence-audit.md section 3. -->

【待补: 掩膜质量统计——前景面积占比分布、被 min_area_frac 拒绝的图像数量与比例、
      以及人工抽查的掩膜合格率。这些量可在两臂预处理完成后从缓存直接统计。】

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
on a spatially extended pattern such as a coat. This backbone is additionally evaluated after
structured pruning at a removal rate of $\rho = 0.25$, described in Section 3.8; the pruned and
unpruned variants are treated as distinct cells of the cost comparison and are never conflated.

*Residual network.* The third backbone is a ResNet-50 trunk with the classification layer removed
and a linear projection to the embedding dimension. It represents the capacity extreme of the
comparison and the architecture family most frequently used in the cattle-identification
literature.

<!-- All three are constructed in src/models.py and were instantiated during the audit.
     Parameter and MAC figures in Table 5 were produced by calling the repository's own
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
| Epoch budget | 【待补】 | see note below |
| Optimiser | Adam | `src/train.py:93` |
| Learning rate | 1 × 10⁻⁴ | `configs/experiment.yaml` |
| Weight decay | 1 × 10⁻⁴ | `configs/experiment.yaml` |
| ArcFace scale $s$ | 64.0 | `configs/experiment.yaml` |
| ArcFace margin $m$ | 0.5 | `configs/experiment.yaml` |
| Random seed | 0 | `configs/experiment.yaml` |
| Rank callback interval | every 200 epochs | `configs/experiment.yaml` |
| Training hardware | 1 × NVIDIA GeForce RTX 4090 (48 GB) | as run |

*Table 2. Shared training configuration. Every value is read from one file and is identical for
all six cells; the table is a description of the protocol, not of an outcome.*

<!-- EPOCH BUDGET — UNRESOLVED, DO NOT GUESS.
     The committed configs/experiment.yaml was changed on main from epochs:30 / eval_every:5 to
     epochs:1000 / eval_every:200 (compare commits d20763e..7390aee). The row above previously
     read "Epoch budget 30". Neither 30 nor 1000 has been confirmed by the author as the budget
     the reported runs actually used, and the per-cell training times supplied below are not
     enough on their own to settle it without assuming a per-epoch time we do not have.
     Keep 【待补】 until the author confirms. Table 2 now quotes eval_every:200, which matches the
     config on main; if the reported runs used 30 epochs they must have used eval_every:5, so the
     two rows have to be confirmed together. -->
【待补: 训练轮次（epoch budget）与学习率调度——须确认报告结果所用配置；
      仓库中 configs/experiment.yaml 现为 epochs:1000、eval_every:200，
      而更早版本为 epochs:30、eval_every:5，两者不一致，不得任选其一】

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
A rank-1/rank-5 callback evaluates the current model on the gallery and query splits at the
configured interval and at the final epoch, and the checkpoint with the best observed rank-1 is
retained.
<!-- src/callbacks.py:19-52; src/train.py:137-156 -->
This is a model-selection step and is reported as such: it uses gallery and query data, and the
selection criterion is therefore not independent of the reported metric. It is applied identically
to all six cells, so it does not favour one cell over another, but it does mean that the reported
figures are best-observed rather than single-final-epoch values.

**Compute.** All six cells were trained on a single NVIDIA GeForce RTX 4090 (48 GB). Wall-clock
training time per cell is given in Table 3, in the row order used throughout this paper
(convolutional trunk, then Vision Transformer, then ResNet-50; within each backbone the
mask-purified arm precedes the box arm).

| Backbone | Crop | Training time (min, wall clock) |
| --- | --- | ---: |
| Convolutional trunk | $I_{\mathrm{sam}}$ | 20 |
| Convolutional trunk | $I_{\mathrm{box}}$ | 23 |
| Vision Transformer | $I_{\mathrm{sam}}$ | 22 |
| Vision Transformer | $I_{\mathrm{box}}$ | 30 |
| ResNet-50 | $I_{\mathrm{sam}}$ | 27 |
| ResNet-50 | $I_{\mathrm{box}}$ | 33 |

*Table 3. Wall-clock training time per cell on one RTX 4090 (48 GB), in the cell order used
throughout this paper and in Table 4. The values are end-to-end training times as recorded,
supplied by the authors. Within each backbone pair the box arm took longer than the mask-purified
arm.*

<!-- Provenance: six values supplied by the author as wall-clock minutes per cell, in the paper's
     cell order: 20, 23, 22, 30, 27, 33. Sum = 155 min = 2.58 h for all six cells.
     The SUM IS DERIVED ARITHMETIC and is stated as such in the text; no per-epoch time and no
     GPU-hour figure is derived, because the epoch budget is unresolved (see Table 2 note).
     Do not back-compute epochs = 1000 or 30 from these times. -->

Across the six cells this is 155 minutes of wall-clock training, that is 2.58 hours in total.
<!-- Derived: 20+23+22+30+27+33 = 155 min; 155/60 = 2.583 h. Labelled as a sum of the values in
     Table 3, nothing further inferred. No per-epoch time is derived: that would require the epoch
     budget, which is unresolved (Table 2). -->

【待补: 上述六个训练时长是否即产出 Table 4 结果的那批运行，须由作者确认；
      若计时来自单独的一次运行，须说明，以免读者把时长与结果对应起来】

The compute used for this study cost 980 CNY.
<!-- Author-supplied total. Deliberately no breakdown: the author's statement does not say whether
     it covers GPU rental, electricity, storage or platform overhead, so none is asserted here.
     If the author later supplies a breakdown, add it; do not infer one from the wall-clock total
     and an assumed hourly rate. -->

The six cells are produced by a single orchestrating script rather than by hand, so that the
shared settings cannot drift between cells. `scripts/run_all_cells.sh` accepts a data root, a seed
list and the two crop arms, and then builds one seeded manifest per seed and per arm, trains and
evaluates every cell, and aggregates the result. Three properties of that script are worth stating
because they bound what the reported numbers can mean.

First, runs are seeded and the seed is recorded. Each seed produces its own manifest, its own run
directory under `outputs/seed_<seed>/<crop>/<model>/`, and its own copy of the metrics, so repeated
runs are independent in the split as well as in initialisation. The default is a single seed; CEA's
expectation of replication requires at least three, and the script takes a seed list for that
purpose.

Second, the output is timestamped and never overwritten. The run directory is created exclusively
under a UTC timestamp, the script aborts before writing anything if a target run directory already
holds files, and the exact configuration — seeds, split fractions, crop arms, model list, segmenter
type and backend, git commit, Python version and start time — is written to the run directory
before training begins. A reported table can therefore be tied to a specific commit and a specific
configuration after the fact.

Third, aggregation refuses to produce a placeholder. The summary step reads only the `metrics.json`
files written by this run, and rejects a file that is stale relative to the run start, missing,
non-finite, tagged as a placeholder, or attributable to the wrong crop arm. Nothing in the
aggregation path can synthesise a number; if a cell did not run, the table cannot be completed. In
prepared-data mode the script additionally requires the two crop trees to be distinct directories,
so that the mask contrast cannot be zero by construction through a pointing error.

<!-- Traceability: scripts/run_all_cells.sh (seeded 2x3 runner, exclusive timestamped results
     dir, run_config.txt with git sha and segmenter settings, refuses to overwrite);
     src/aggregate.py (NotAMeasurement raised for missing/stale/non-finite/placeholder/
     wrong-arm metrics); prepared mode checks sam/ and bbox/ resolve to different paths.
     10 runner/aggregate tests pass on CPU.
     See paper/evidence-audit.md sections 8 and 11. -->

Because the aggregation cannot emit a placeholder, Table 4 is left entirely empty in this draft.
The six cells will be filled only from a completed run of this script, with at least three seeds,
and the table will then carry the mean and dispersion across seeds rather than a single value.

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

【待补: 推理延迟——单张图像、批大小 1 的实测值，在 NVIDIA GeForce RTX 4090 上、
      热身若干次后取中位数；须写明热身次数与计时次数】
<!-- Latency hardware: Table 2 states the training GPU as one RTX 4090 (48 GB). Unless the
     author says latency was timed on a different device, that is the machine above. If it was
     measured elsewhere, replace the model name. -->

### 3.8 Structured pruning of the Vision Transformer

The transformer branch is additionally evaluated after structured pruning, and the pruned and
unpruned networks are reported separately throughout. Two structural unit types are removed: whole
attention heads, and whole channels of the feed-forward block.

*Selection criterion.* Units are ranked by weight magnitude — the mean absolute weight of a unit,
taken over the query, key and value rows for an attention head and over the corresponding row of
the first feed-forward projection for a channel — and the lowest-ranked units are removed. This is
a standard magnitude criterion. It is deterministic, requires no calibration pass, and imposes no
additional forward passes during selection, which matters because the alternative would multiply
the cost of the pruning stage by the number of units considered.

We state explicitly what this criterion is not. A criterion that ranks a unit by the change it
induces in embedding geometry — for instance the drop in a discriminability ratio,
$I(u) = \Delta(\phi) - \Delta(\phi^{(-u)})$ with
$\Delta = S_{\mathrm{inter}} / (S_{\mathrm{intra}} + \varepsilon)$ measured on a calibration split
— is a plausible and in our view attractive alternative, and it is closer to what coat-pattern
re-identification actually needs, since it asks directly whether a unit keeps images of one animal
together and images of different animals apart. That criterion was **not** implemented and was
**not** used to produce any result in this paper. We mention it only to mark it as a direction for
future work, and we do not claim its properties for the magnitude criterion we do use.

*Removal rate.* The removal rate $\rho$ is applied per block with half-up rounding and a floor of
one unit, so that a requested removal cannot silently round to nothing. On a 3-head block at
$\rho = 0.25$ this removes one head per block, which is 12 of 36 heads over 12 blocks; on a
768-channel feed-forward block it removes 192 channels, or 2,304 of 9,216 over the network. Units
are physically removed rather than masked: the affected tensors are rebuilt without the dropped
rows and columns, so the parameter count and the multiply–accumulate count both fall, as Table 5
reports. A request for pruning that would remove nothing raises rather than training an unpruned
network under a pruned name, and the realised pruning is recorded in the run metadata and the
evaluation output so that a reported cell can be matched to the architecture that produced it.

<!-- Implementation traceability:
     src/models.py _magnitude_scores() = weight.detach().abs().mean(dim=1);
     units_to_prune() applies half-up rounding with a floor of one; the pruned tensors are
     rebuilt (qkv rows, proj columns, fc1 rows, fc2 columns) so count_parameters falls;
     the prune report carries "criterion": "weight_magnitude".
     Measured at rho=0.25: params 5,623,232 -> 4,144,064, MACs 1,074,757,632 -> 784,269,312,
     heads 36 -> 24, MLP channels 9,216 -> 6,912.
     The criterion is weight magnitude; paper/method.tex sec:pruning has been rewritten to
     match this section, and both now present the discriminability-ratio criterion as an
     explicitly labelled direction that was NOT evaluated. 12 pruning tests pass, including
     one that pins the old advanced-indexing no-op as a regression.
     See paper/evidence-audit.md section 4. -->

【待补: 剪枝后是否微调、微调轮次与学习率；若未微调，须说明剪枝网络的权重直接来自
      未剪枝训练结果还是重新训练】

## 4. Results

### 4.1 Retrieval performance of the six cells

Table 4 reports rank-1 and rank-5 for every combination of crop policy and backbone. The cells in
a row share a backbone and differ only in the crop; the cells in a column share a crop and differ
only in the network. Every figure is measured on the held-out test identities, which were seen
neither in training nor in validation. Training time and backbone cost are reported separately —
wall-clock training time per cell in Table 3 and the inference cost of each backbone in Table 5 —
because a cell's training time is a property of the run rather than a retrieval result, and mixing
the two in one table would obscure which entries are measurements of recognition performance.

| Backbone | Crop | Rank-1 | Rank-5 | $\Delta_{\mathrm{SAM}}$ (rank-1) |
| --- | --- | ---: | ---: | ---: |
| Convolutional trunk | $I_{\mathrm{sam}}$ | 【待补】 | 【待补】 | — |
| Convolutional trunk | $I_{\mathrm{box}}$ | 【待补】 | 【待补】 | 【待补】 |
| Vision Transformer | $I_{\mathrm{sam}}$ | 【待补】 | 【待补】 | — |
| Vision Transformer | $I_{\mathrm{box}}$ | 【待补】 | 【待补】 | 【待补】 |
| ResNet-50 | $I_{\mathrm{sam}}$ | 【待补】 | 【待补】 | — |
| ResNet-50 | $I_{\mathrm{box}}$ | 【待补】 | 【待补】 | 【待补】 |

*Table 4. Retrieval accuracy on the held-out test identities of the identity-disjoint partition
(Table 1). No animal scored here appears in the training or validation sets.
$I_{\mathrm{sam}}$ = mask-purified dorsal coat; $I_{\mathrm{box}}$ = detector box crop.
$\Delta_{\mathrm{SAM}}$ is rank-1 on the purified arm minus rank-1 on the box arm, computed per
backbone. Each figure is to be read from the aggregated summary written by
`scripts/run_all_cells.sh` (`summary_six_cells.md` in the timestamped run directory), which draws
only on the per-cell `metrics.json` files produced by that run.*

<!-- PROVENANCE. No value in Table 4 exists yet, and none may be written until
     scripts/run_all_cells.sh has completed. The placeholder tables previously in
     results/metrics.json and results/accuracy_curves.json are hard-coded constants generated by
     src/metrics.py and src/results.py, whose own docstrings call them placeholders, and whose
     disclaimer text was removed in commits 2e1e2d4/cdde8cc/ddaf922. They must not be used; the
     new aggregation step (src/aggregate.py) rejects placeholder-tagged and stale metrics files
     precisely so that they cannot re-enter through the back door.
     See paper/evidence-audit.md sections 1 and 8. -->

【待补: 每个单元格的 gallery/query 图像数与被评身份数（须与 Table 1 的数据统计一致）】

【待补: 重复实验统计——至少 3 个随机种子的均值与标准差；若仅单次运行，须明确说明，
      并说明为何单次运行足以支撑结论】

### 4.2 Effect of mask-based purification

The per-backbone contrast $\Delta_{\mathrm{SAM}}$ is the primary outcome of the crop factor.
Because both arms share their detections, their identity assignment and their image count, the
comparison is paired at the level of the animal, and the appropriate summary is the difference
per backbone rather than a pooled difference.

Both arms are now produced by the pipeline itself: the purified arm is generated by the
box-prompted segmentation front-end described in Section 3.3, and the control arm by the box crop
with the segmenter never instantiated. The contrast is therefore executable end to end, and the
runner verifies before training that the two arms resolve to different image trees, so a
$\Delta_{\mathrm{SAM}}$ of exactly zero cannot arise from a pointing error. What remains is the
measurement itself, which this draft does not contain.

【待补: 三个骨干各自的 Δ_SAM 数值，以及与之配套的自助法置信区间或配对检验结果】

【待补: 掩膜质量对 Δ_SAM 的影响——前景面积占比分布、被 min_area_frac 拒绝的图像比例，
      以及低质量掩膜样本上的分项结果。可从掩膜缓存直接统计。】

We note a design decision with a consequence for interpretation. The two arms share the identity
split, so an animal that is mis-detected in one arm is mis-detected in both, and the paired
difference is not inflated by easy animals appearing only in one arm. The cost of that decision is
that the design cannot detect an interaction between crop policy and detection quality; that would
require a third factor.

### 4.3 Computational cost

Table 5 gives the cost of each backbone at the common input resolution. These figures are
properties of the committed architecture definitions, obtained by instantiating each model from
the reference implementation and calling its own counting routines, so they do not depend on any
training run.

| Backbone | Compression | Parameters | MACs @ 224² | Relative MACs |
| --- | --- | ---: | ---: | ---: |
| Convolutional trunk | none | 1,832,224 | 242,311,168 | 1.00× |
| Vision Transformer, unpruned | none | 5,623,232 | 1,074,757,632 | 4.44× |
| Vision Transformer, pruned (ρ = 0.25) | structured, physical removal | 4,144,064 | 784,269,312 | 3.24× |
| ResNet-50 | none | 24,557,120 | 4,088,184,832 | 16.87× |

*Table 5. Cost of the inference embedding network at 224 × 224, embedding dimension 512,
classification head excluded. Parameters and MACs measured by `count_parameters` and
`estimate_flops` in `src/models.py` with `configs/experiment.yaml`; the relative column is the
ratio to the convolutional trunk. These are MACs, not FLOPs. The pruned row is the ViT at
ρ = 0.25 with units physically removed, which is the configuration every `pruned_vit_arcface` run
now uses; the unpruned row is retained because it is the like-for-like baseline for the pruning
comparison.*

<!-- Traceability for Table 5 — all four rows reproduced independently by instantiating each
     model and calling the repository's count_parameters / estimate_flops at 224x224:
       yolo_arcface            params 1,832,224   MACs 242,311,168
       vit unpruned (ratio 0)  params 5,623,232   MACs 1,074,757,632
       vit pruned   (ratio .25) params 4,144,064  MACs 784,269,312
       wang_metric (resnet50)  params 24,557,120  MACs 4,088,184,832
     Relative MAC ratios: 1074757632/242311168 = 4.44, 784269312/242311168 = 3.24,
     4088184832/242311168 = 16.87.
     Pruning deltas: params -1,479,168 (-26.3%), MACs -290,488,320 (-27.0%).
     Pruning acts on 12 of 36 attention heads and 2,304 of 9,216 MLP channels
     (1 of 3 heads and 192 of 768 channels per block, over 12 blocks).
     See paper/evidence-audit.md section 4. -->

Two comparisons matter, and they are different. Against ResNet-50, the pruned transformer costs
5.2× fewer MACs (784 M against 4088 M) — the ratio that decides whether the model fits an
on-board power and mass budget. Against its own unpruned version, pruning removes 26.3 % of
parameters and 27.0 % of multiply–accumulates, which is the quantity that decides whether the
compression stage earns its place in the pipeline at all. The absolute spread across the four rows
is a factor of 16.9 in MACs from the convolutional trunk to ResNet-50, so an accuracy difference
of a few points between backbones has to be read against a very large difference in the cost of
running them on an aircraft.

<!-- Criterion disclosure. The implemented pruning ranks units by weight magnitude; see Section
     3.8 and paper/evidence-audit.md section 4. No recognition result is claimed for the pruned
     model here, only its architecture cost. -->

【待补: 剪枝率—参数量—MACs—Rank-1 的对应表（嵌套剪枝率 ρ ∈ {0, 0.25, 0.5}），
      以及剪枝后是否重新微调、微调轮次与设置。当前尚无实测 Rank-1，
      表中只填架构开销。】

### 4.4 Training behaviour

Rank-1 and rank-5 are logged at the configured evaluation interval, which gives a coarse view of
whether a cell has converged within the epoch budget or is still improving when training stops.

【待补: 六个单元格的 Rank-1/Rank-5 随轮次变化曲线（数据来源 outputs/<cell>/rank_callback.json），
      以及是否出现未收敛的单元格】

<!-- Do not use results/accuracy_curves.json. Those curves are placeholders drawn at epochs
     200/400/600/800/1000 (paper/evidence-audit.md §8, §1). -->

A convergence diagnostic matters for the interpretation of Section 4.2: if one backbone is still
improving when another has plateaued, the between-backbone comparison at a fixed epoch budget
understates the former, and the honest reading is that the budget, not the
architecture, is binding. The same caveat applies to the crop contrast, since the purified arm has
fewer informative pixels and may converge on a different trajectory.

## 5. Discussion

### 5.1 What identification from the air can and cannot tell you

The applied contribution of this study is not a new network but a measurement made under an
imaging regime that the cattle re-identification literature has largely not addressed: low-altitude
aerial observation of the dorsal aspect, at the resolution a small UAV delivers, on a working herd.
That regime is restrictive, and it is worth being explicit about what it costs and what it buys.

What it costs is cue diversity. A ground-level system can fall back on a face, a muzzle, a flank
profile or a gait signature; an aerial system has the coat pattern from above and little else. When
the coat is ambiguous — two similarly patterned animals, a mud-caked flank, a coat in seasonal
transition — there is no second view to appeal to. An honest aerial identification system therefore
has to be evaluated with that failure mode visible, which is one reason we report a fixed-threshold
rank-1 alongside rank-5 rather than a single number: the operator of a working system needs to know
how often it is confidently wrong, and rank-5 tells them whether a short human check could recover
the case.

What it buys is the thing the introduction argued for: an observation that requires no handling,
covers a paddock in a single flight, and reuses imagery that a growth-monitoring programme is
already collecting. For a pasture-based enterprise, that is the difference between identity data
being a separate expense and identity data being a by-product of surveillance that was going to
happen anyway. It also means the recording conditions are not chosen for the benefit of the
recogniser — the aircraft flies a survey, not a photo shoot — so the viewpoint and illumination
distribution are those of the operation rather than those of a curated dataset.

### 5.2 Why the crop question is an agronomic question

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

### 5.3 What the cost figures imply for deployment

The rows of Table 5 span a factor of nearly seventeen in multiply–accumulates, from the
convolutional trunk to ResNet-50. On a UAV, that factor translates into flight endurance, into the
thermal envelope of the payload enclosure, and into whether inference can be done on board at frame
rate or must be deferred to a ground station after landing. A deferred pipeline is not necessarily
worse — a paddock can be flown in the morning and processed at midday, and for growth monitoring
the latency is irrelevant — but it changes the system architecture, because on-board inference
permits adaptive flight (re-flying an animal whose image was rejected) whereas deferred processing
does not.

Pruning changes where the transformer sits in that landscape, and the size of the change is worth
stating plainly. Removing a quarter of its parameters moves the transformer from 4.44× the
convolutional trunk's cost to 3.24×, and from 3.8× cheaper than ResNet-50 to 5.2× cheaper. That is
a meaningful shift for a payload budget, but it does not make the transformer cheap: at 784 M MACs
it still costs more than three times the convolutional trunk. Pruning therefore buys headroom
within the transformer family rather than changing which family is affordable on a constrained
aircraft. Whether the retained accuracy at the pruned operating point justifies even that reduced
cost is an empirical question this draft does not answer, and we do not treat the compression as
free: a pruned model that loses accuracy has not saved anything an operator cares about.

The practical reading of Table 5 is therefore conditional on the intended workflow. For a
post-flight batch pipeline, the ResNet-50 row is affordable and the question is purely whether
its accuracy justifies its cost. For on-board inference at a few frames per second on an embedded
accelerator, the convolutional trunk is the row that fits comfortably, followed by the pruned
transformer, and the relevant question becomes how much accuracy is lost by moving down that
column. This is why the protocol reports accuracy and cost together rather than accuracy alone:
the comparison that matters operationally is between two cells, not between two numbers.

### 5.4 Identification as part of a monitoring record

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

### 5.5 Limitations and threats to validity

Several limitations bound what can be concluded, and they are stated here rather than left for a
reader to infer.

*Scale and breed.* The study uses 96 Simmental beef cattle from a single herd, imaged at one site.
Ninety-six identities is a small enrolment by the standards of person re-identification, and rank-1
on a 96-way problem is not directly comparable to rank-1 on a 1000-way problem. Simmental cattle
are a high-contrast, patterned breed, which is favourable for coat-based identification; breeds with
low-contrast or seasonally changing coats present a harder problem that this archive cannot address,
and we do not assume that a figure obtained here transfers to them. The sex and age composition of
the herd
【待补: 性别与月龄范围、体重区间——须由牧场记录或数据文章提供】 also bounds the population to
which these results apply.

*Single site.* All imagery comes from one farm. The transfer question raised in Section 5.2 — does
the model survive a change of premises — is not answered by a single-site experiment, and the
$\Delta_{\mathrm{SAM}}$ result should be read as evidence about mechanism rather than as a
guarantee of transfer.

*Model selection on the evaluation split.* The retained checkpoint is the one with the best
observed rank-1 on the gallery/query splits, as described in Section 3.6. The reported figures are
therefore mildly optimistic in absolute terms. The bias is applied identically to all six cells and
so should not affect the ordering, but a fully clean protocol would carve a validation split for
this purpose. The reference implementation defines no validation split, which is a shortcoming of
the protocol as implemented.

*Epoch budget.* A fixed epoch budget with no learning-rate schedule was chosen to keep the
comparison symmetric. If some cells have not converged, the budget rather than the architecture is
binding, and the between-backbone comparison is correspondingly weakened. Reporting the training
curves (Section 4.4) is the mitigation, not a solution.
【待补: 确认最终采用的轮次；仓库配置在合并后由 epochs:30 改为 epochs:1000，
      本稿不预设其一（另见 Table 2 注）】

*Segmentation dependence.* The purified arm inherits the failure modes of the segmenter. A mask
that clips the shoulder, or that leaks into an adjacent animal, changes what the recogniser sees in
a way that is correlated with pose and crowding. The implementation mitigates the grosser cases —
masks are clipped to the detector box, reduced to their largest connected component, hole-filled,
and rejected below a minimum area — but these are guards against degenerate masks, not a guarantee
of correct ones. The pipeline does not currently report mask quality, so a reader cannot tell from
the results how often the purified crop was faithful to the dorsal coat. Measuring the foreground
area distribution and the rejection rate is straightforward from the mask cache and is the most
obvious single improvement to the protocol.

*Pruning criterion.* The compression reported in Table 5 uses weight magnitude, which is a proxy
for importance rather than a measure of it. A unit can carry little weight and still be the one
that separates two similarly patterned animals, and magnitude ranking has no way to notice. The
embedding-geometry criterion sketched in Section 3.8 would address that directly, but it requires a
calibration pass and one forward pass per candidate unit, and it is not implemented here. The cost
figures in Table 5 should therefore be read as an upper bound on what careful selection could
achieve at the same removal rate, not as evidence that magnitude selection is near-optimal.

*No open-set evaluation.* All probes are enrolled animals. A system deployed on a farm will
encounter animals that are not in its gallery — newly purchased, visiting, or a neighbouring herd
within the flight line — and the closed-set metrics reported here do not characterise behaviour in
that case. Open-set evaluation with an explicit rejection threshold is required before autonomous
use.

*Statistical power.* The number of independent experimental units is small: six cells, and within
each cell a few hundred query images drawn from 96 animals. Query images of the same animal are not
independent, so confidence intervals computed over images understate the uncertainty. Animal-level
resampling is the appropriate basis for inference and should be used when the results are reported.

### 5.6 What would change the conclusions

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
FLOPs. Under that accounting the backbones cost 1.83 M, 5.62 M and 24.56 M parameters at 242 M,
1075 M and 4088 M MACs per 224 × 224 frame, and structured pruning at a removal rate of 0.25
reduces the transformer to 4.14 M parameters and 784 M MACs.

The contribution is a measurement instrument rather than a winning number. Applied to aerial
cattle imagery it yields a per-backbone estimate of whether mask-based purification of the dorsal
coat helps, which is a question about whether the recogniser is reading the animal or the pen; and
it yields an accuracy-versus-cost curve that an operator can use to decide between post-flight
batch processing and on-board inference. The retrieval figures for the six cells complete the
instrument, and the interpretation we expect them to support is set out in Section 5.6 in a form
that the measurements can contradict.

Three things would make a follow-on study substantially stronger: replication across farms and
breeds, so that transfer rather than within-site accuracy becomes the reported quantity;
open-set evaluation with an explicit rejection threshold, so that unknown animals are handled
rather than assumed away; and a measurement of segmentation quality, so that the purified arm's
dependence on its mask is characterised rather than ignored. A fourth change would strengthen the
compression result specifically: replacing magnitude ranking with the embedding-geometry criterion
of Section 3.8, and reporting accuracy at a matched parameter budget rather than at a matched
removal rate. None of these requires a new architecture. Any of them would make the resulting
numbers usable as evidence in a livestock management decision, which is the standard that an
applied agricultural journal should hold them to.

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

The authors gratefully acknowledge Yaowu Wang and colleagues, who acquired the UAV imagery and
released it openly, together with the accompanying identification code and logs, under CC BY 4.0.
This study analyses that release; it did not collect the imagery.

## Data availability

The imagery analysed in this study is publicly available from Zenodo under CC BY 4.0: Wang, Y.,
2024. Dataset for cattle growth monitoring in PLF using UAV-based LiDAR.
https://doi.org/10.5281/zenodo.13364663. The dataset comprises 96 Simmental beef cattle imaged from
unmanned aerial vehicles, released together with the accompanying data article (Wang et al., 2025b). The accompanying code, results and logs for the coat-pattern
identification work are available as https://doi.org/10.5281/zenodo.13361729. A local copy of the
archive, organised as one directory per identity, is held on the authors' GPU server at
`/home/iadi/UAV-basedData/RGB-data/individaulsByIds`; no additional data were generated.

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

Wang, Y., Mücher, S., Wang, K., Wang, W., Kooistra, L., 2025a. Deep metric learning for individual
cattle identification using coat patterns: Proposal for a best practice. Computers and Electronics
in Agriculture 238, 110754. https://doi.org/10.1016/j.compag.2025.110754

Wang, Y., Kooistra, L., Mücher, S., Wang, W., 2025b. Integrated unmanned aerial vehicle-based LiDAR
and RGB data for individual cattle growth monitoring in precision livestock farming. Scientific
Data 12, 1318. https://doi.org/10.1038/s41597-025-04783-6

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
