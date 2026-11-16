# Individual cattle identification from UAV-RGB images organized by identity

**Authors**

Tianyi Ma^a,^\*

^a^ School of Computer and Artificial Intelligence, Zhengzhou University, Zhengzhou 450001, China

\* Corresponding author. E-mail: pending; ORCID: [0000-0003-1167-621X](https://orcid.org/0000-0003-1167-621X)

## Abstract

Individual identification from imagery lets a herd be managed as animals rather than as a group. Wang et al. (2025a) showed that dorsal coat pattern supports this task, and they released a deep-metric-learning practice built on an open UAV-RGB collection of 96 cattle. This study uses that same UAV-RGB herd. Every experiment reads those images as stored, one folder per identity. No SAM mask and no detector crop are built. Three embedding networks are trained under one ArcFace objective and one identity-disjoint partition, so that a correct match cannot be a near-duplicate frame of an animal already seen in training. A further group adds farm-aware optimal dataset selection (F-ODS) and farm-knowledge-guided self-supervised dynamic learning. Rank-1, rank-5, closed-set accuracy and mean average precision are pending the rerun on NVIDIA GeForce RTX 4090(48GB).

**Keywords:** cattle identification; coat pattern; unmanned aerial vehicle; ArcFace; identity-disjoint evaluation

## 1. Introduction

Precision livestock farming needs a record that stays attached to one animal. Ear tags and collars do that, at the cost of handling. Coat pattern is a biometric that can be read from images already collected for monitoring (Andrew et al., 2021; Wang et al., 2025a). From the ground the flank fills the frame. From a UAV the animal is small and seen from above, and the usable cue is the dorsal coat.

Wang et al. (2025a) treated that cue directly. They trained a deep metric learner on the open UAV-RGB set of 96 Simmental males and checked it on a ground-RGB set of the same herd and on an external Holstein set. Their published figures were obtained under their mining objective and their own split. They are not results of the protocol below.

This paper asks a narrower question on the same UAV-RGB images. The comparison does not build a SAM mask or a detector crop. Three networks that differ in capacity share the loss, the resolution and an identity-disjoint split. F-ODS and a quality-gated self-training stage are then added as a separate group. All recognition numbers are pending the rerun on NVIDIA GeForce RTX 4090(48GB). No figure from an image-level split is used.

## 2. Relation to coat-pattern identification

Coat-pattern re-identification was established on ground imagery of Holstein cattle by metric learning (Andrew et al., 2021). Wang et al. (2025a) moved the same idea to the dorsal coat of the 96-animal UAV-RGB herd used here, and they set out a training practice rather than a new backbone: how to choose the model and the hyperparameters, and what semi-hard mining changes. That practice is the predecessor of this study. We do not repeat it. We hold the loss fixed, we change the trunk, and we keep every test identity out of training.

The loss is the additive angular margin of Deng et al. (2019). The three trunks are a small convolutional network, a pruned Vision Transformer (Dosovitskiy et al., 2021) and a ResNet-50 (He et al., 2016). The ResNet stands in for the capacity of the coat-pattern networks used by Wang et al. (2025a). It is trained from random initialization under ArcFace. It is not their released model, and their scores are not entered in our tables.

The images are the UAV-RGB files as released, one directory per animal. This study does not mask them and does not crop them again.

## 3. Materials and methods

### 3.1 Inputs and study design

Let $x$ be a dorsal-coat patch and $y(x)$ its identity when known. A model is an embedding $\phi_\theta$ trained so that the cosine of $\ell_2$-normalized embeddings is large within an identity and small across identities. At test time the ArcFace matrix is discarded.

Every $x$ is one UAV-RGB image from an identity folder. The pipeline starts at that file. It does not run a segmenter and it does not run a detector.

The present experiment has two blocks, both scored on the same test identities.

1. Three variants, trained with uniform weights on all training images of the training identities: the convolutional trunk, the pruned Vision Transformer, and the ResNet-50.
2. The same three variants under F-ODS, under farm-knowledge-guided dynamic learning, and under both.

### 3.2 Data acquisition

The images are the UAV-RGB subset of the 96-individual herd described by Wang et al. (2025a): semi-free-range Simmental male beef cattle, the same animals in their UAV-RGB and ground-RGB collections. This study uses the UAV-RGB collection only. The files are the public archive of Wang (2024), https://doi.org/10.5281/zenodo.13364663, released under CC BY 4.0. A working copy is held on NVIDIA GeForce RTX 4090(48GB).

Flight altitude, ground sampling distance and camera metadata are reported by Wang et al. (2025a). They are not repeated here, and no sensor or flight figure is introduced in this manuscript. Breed, sex and herd size are taken from that paper: 96 Simmental males.

### 3.3 Identity-disjoint partition

Consecutive UAV frames of one animal are near-duplicates. An image-level split would place those frames on both sides of the train–test cut and would inflate rank-1. The partition is therefore by identity. Each animal is assigned to exactly one of training, validation and test, and every patch of that animal follows it.

Identity identifiers are sorted and shuffled with a fixed seed. Whole animals are then assigned at the proportions two thirds, one sixth and one sixth, by largest remainder. An animal enters the test set only if it has at least two patches, one for the gallery and one for the query. Gallery and query are both drawn from the test animals. Validation animals are used to choose F-ODS coefficients, the dynamic-learning gates and the checkpoint. They are not used as test queries.

Before training or evaluation, the three identity sets are checked for overlap. Any shared identity aborts the run. The test set is read only at final evaluation. Counts of identities and patches in each set, and any animal excluded by the two-patch rule, are pending; they will be copied from the manifest of the NVIDIA GeForce RTX 4090(48GB) rerun. Results from any earlier within-identity split are discarded and are not reported.

### 3.4 Recognition variants

All three variants take a $224 \times 224$ patch and emit a 512-dimensional embedding. None is initialized from a pretrained checkpoint.

The convolutional variant is a five-stage stack of convolution, batch normalization and SiLU, with stride-2 downsampling, global average pooling and a linear map to the embedding. The name YOLO-ReID in the code is historical. The trunk does not localize animals.

The transformer variant is a patch-16 Vision Transformer with 12 blocks, 3 heads and a token width of 192 (Dosovitskiy et al., 2021). Before training, whole attention heads and whole feed-forward channels are removed by weight magnitude at rate $\rho = 0.25$, with at least one unit removed per block. The tensors are rebuilt without the dropped rows and columns, and the pruned network is trained from scratch. Ranking units by the drop in an embedding discriminability ratio was considered and is not used (Section 3.8).

The residual variant is a ResNet-50 with the classification layer replaced by a linear map to the embedding (He et al., 2016). It is the high-capacity reference, not a copy of Wang et al. (2025a).

### 3.5 Shared objective and training

For a normalized embedding $z_i$ and normalized class weights, with $\theta_{y_i}$ the angle to the true class, scale $s$ and margin $m$,

$$\ell_i = -\log \frac{e^{s\cos(\theta_{y_i}+m)}}{e^{s\cos(\theta_{y_i}+m)}+\sum_{j\neq y_i}e^{s\cos\theta_j}}.$$

The constants $s = 64$ and $m = 0.5$ are shared and are not tuned per variant (Deng et al., 2019). On the three-variant comparison each patch has weight one. The F-ODS arm replaces that weight by the quality $Q$ of Section 3.6 and does not change the logits.

Training uses Adam, learning rate $10^{-4}$, weight decay $10^{-4}$, batch size 8 and 1000 epochs, with no schedule and no early stopping. Augmentation is resizing, horizontal reflection and mild colour jitter. Metrics are computed every 200 epochs, at epochs 200, 400, 600, 800 and 1000. The checkpoint with the best rank-1 on the validation identities is the one scored on the test identities. The rerun is on NVIDIA GeForce RTX 4090(48GB). Wall-clock time is pending.

Closed-set rank-$k$ is the fraction of test queries whose true identity is among the $k$ nearest distinct gallery identities, by cosine similarity. Closed-set accuracy is rank-1 under that ranking. Mean average precision is the mean, over test queries, of the average precision of the same ranking. Parameter and multiply–accumulate counts are taken on $\phi_\theta$ at $224 \times 224$, excluding the ArcFace matrix, and are pending with the recognition metrics.

### 3.6 Farm-aware optimal dataset selection

UAV sets are long-tailed: animals under the flight path contribute many patches, and animals at the edge, lying, or blurred contribute few. F-ODS selects a smaller training support. It does not remove test patches, and a low-quality test patch is still scored.

Five scores in $[0,1]$ are computed on each training image $x$. $F(x)$ is that image and $\Omega$ is its frame. No mask is available, so the score uses the stored pixels. Percentiles and the reference aspect ratio are fit on training identities only.

Body visibility $V$ penalises padding, truncation at the patch border and a second animal. With $T(x)$ the part of $F(x)$ that touches the border and $O(x)$ the part of the patch belonging to another animal,

$$V(x)=\frac{|F(x)|}{|\Omega|}\left(1-\frac{|T(x)|}{|F(x)|}\right)\left(1-\frac{|O(x)|}{|\Omega|}\right).$$

Coat cleanliness $C$ is high when the Laplacian response $\kappa$ inside $F(x)$ is high and the fraction $\eta$ of near-black or near-white pixels is low. With training-pool percentiles $\kappa_5$ and $\kappa_{95}$,

$$C(x)=\mathrm{clip}_{[0,1]}\!\left(\frac{\kappa(x)-\kappa_5}{\kappa_{95}-\kappa_5}\right)(1-\eta(x)).$$

Viewing angle $A$ prefers a compact, near-nadir back. With solidity $\sigma$, width-to-height ratio $\rho$ and reference ratio $\rho_0$,

$$A(x)=\sigma(x)\exp\!\left(-\left|\log\frac{\rho(x)}{\rho_0}\right|\right).$$

Illumination $L$ prefers mid-grey, low-variance luminance. With mean $\mu$, standard deviation $\varsigma$ and the training-pool percentile $\varsigma_{95}$,

$$L(x)=\mathrm{clip}_{[0,1]}(1-2|\mu(x)-0.5|)\cdot\mathrm{clip}_{[0,1]}\!\left(1-\frac{\varsigma(x)}{\varsigma_{95}}\right).$$

Sharpness $S$ is the variance $v$ of the Laplacian inside $F(x)$, scaled by its training-pool percentiles $v_5$ and $v_{95}$:

$$S(x)=\mathrm{clip}_{[0,1]}\!\left(\frac{v(x)-v_5}{v_{95}-v_5}\right).$$

The quality score is

$$Q=\alpha V+\beta C+\gamma A+\delta L+\epsilon S,$$

with non-negative weights summing to one, so $Q\in[0,1]$. The weights are not identity labels and are not fit on the test identities. Their purpose is to keep patches in which the coat is visible, clean, seen from above, evenly lit and sharp, and to stop frequent identities from dominating the loss.

For $N$ identities and a per-identity count $m$, the support set keeps the $m$ highest-$Q$ training patches of each identity,

$$D(N,m)=\{x_{ij}\mid i=1,\ldots,N,\ j=1,\ldots,m\},$$

where $x_{ij}$ ranks $j$-th by $Q$ within identity $i$. An identity with fewer than $m$ patches contributes only what it has. It is not duplicated.

A candidate support set is scored by

$$\max\left(\mathrm{Acc}(D)-\lambda|D|+\mu Q(D)\right),$$

where $\mathrm{Acc}(D)$ is rank-1 on the validation identities, $|D|$ is the number of selected patches, $Q(D)$ is the mean quality, and $\lambda>0$, $\mu>0$. The operating point is

$$(N^{\star},m^{\star})=\arg\max_{N,m}\mathrm{Acc}\!\left(D(N,m)\right).$$

$N^{\star}$ does not drop test identities. Every test identity remains in the evaluation. The coefficients, $\lambda$, $\mu$ and $(N^{\star},m^{\star})$ are chosen on the validation identities and then frozen. Their realised values are pending.

In the F-ODS arm the loss weight of a selected patch is $w_i=Q(x_i)$. Identities are sampled uniformly, then a patch inside the identity is sampled in proportion to $w_i$. The control arm uses every training patch of the same identities with $w_i=1$. The test patches are the same in both arms.

### 3.7 Farm-knowledge-guided self-supervised dynamic learning

This stage admits further images from the same identity-folder collection that have no label the loss may trust. “Self-supervised” means self-training: the current model proposes the identity, and $Q$ decides whether the proposal may enter training. The ArcFace loss is not replaced by a contrastive pretext task.

The unlabeled pool $\mathcal{U}$ excludes every patch that can be linked to a validation or test identity, including other frames from the same flight sequence as a held-out animal. If that exclusion leaves $\mathcal{U}$ empty, the arm is reported as not run. It is not filled with held-out animals. $Q$ on $\mathcal{U}$ uses the normalizers and coefficients already frozen in Section 3.6.

Each remaining patch is compared with prototypes built from the labeled training support only. Confidence and the pseudo-label are $c(x)=\max_y s(x,y)$ and $\hat{y}(x)=\arg\max_y s(x,y)$, with $s$ the cosine to the prototype of a training identity. A patch is admitted only if

$$c(x)\ge\tau_c\quad\text{and}\quad Q(x)\ge\tau_q.$$

Both gates are set on the validation identities before the test identities are read. An admitted patch is appended with weight $w=Q(x)$. Every $R$ updates, $\mathcal{U}$ is re-embedded and the admitted set is rebuilt. A patch that falls below either gate leaves the pool. Prototypes used for the next admission test come from the labeled support only. After the last round the test identities are scored once. No admitted patch is used as a query, a gallery image or a prototype component. $\tau_c$, $\tau_q$, $R$ and the number of admitted patches are pending.

### 3.8 Stages not implemented

**In the present experiment, measurements pending on NVIDIA GeForce RTX 4090(48GB).** The three variants with uniform weights. F-ODS, including $Q$, the selection objective, $(N^{\star},m^{\star})$ and the weight $w_i=Q(x_i)$. Farm-knowledge-guided dynamic learning, including the two gates. These rows are the tables in Section 4. A row in the protocol is not a measured result.

**Not in this study.** A SAM mask and a detector crop. The images are used as stored, so neither arm is trained. Pruning by embedding discriminability. The transformer uses weight magnitude; the geometry criterion is future work and has no row. Any image-level or within-identity split. That partition is excluded, not deferred. Published scores from Wang et al. (2025a). They were produced under a different objective and are not written into Section 4.

## 4. Results

No cell below has been filled. Each entry waits on the identity-disjoint rerun on NVIDIA GeForce RTX 4090(48GB). Placeholder curves and tables from earlier drafts are not a source.

### 4.1 Three variants

Table 1 is the uniform-weight comparison. Table 2 is closed-set accuracy at the five evaluation epochs and is the source of the learning-curve figure, which is not plotted until the logs exist.

| Variant | Rank-1 | Rank-5 | Accuracy | mAP | Parameters | MACs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Convolutional | pending | pending | pending | pending | pending | pending |
| Pruned Vision Transformer | pending | pending | pending | pending | pending | pending |
| ResNet-50 | pending | pending | pending | pending | pending | pending |

*Table 1. Three variants on the identity-disjoint test identities. Images are used as stored. Uniform weights, no F-ODS, no dynamic learning. Pending the rerun on NVIDIA GeForce RTX 4090(48GB).*

| Variant | 200 | 400 | 600 | 800 | 1000 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Convolutional | pending | pending | pending | pending | pending |
| Pruned Vision Transformer | pending | pending | pending | pending | pending |
| ResNet-50 | pending | pending | pending | pending | pending |

*Table 2. Closed-set accuracy at epochs 200, 400, 600, 800 and 1000. Pending the rerun on NVIDIA GeForce RTX 4090(48GB).*

| Variant | Wall-clock training time (min) |
| --- | ---: |
| Convolutional | pending |
| Pruned Vision Transformer | pending |
| ResNet-50 | pending |

*Table 3. Wall-clock minutes on NVIDIA GeForce RTX 4090(48GB) for the three runs in Table 1. Pending.*

### 4.2 F-ODS and dynamic learning

Table 4 crosses the three variants with four training pools. “Uniform” repeats Table 1 and is the control. The test identities do not change.

| Training pool | Variant | Rank-1 | Rank-5 | Accuracy | mAP |
| --- | --- | ---: | ---: | ---: | ---: |
| Uniform | Convolutional | pending | pending | pending | pending |
| Uniform | Pruned Vision Transformer | pending | pending | pending | pending |
| Uniform | ResNet-50 | pending | pending | pending | pending |
| F-ODS | Convolutional | pending | pending | pending | pending |
| F-ODS | Pruned Vision Transformer | pending | pending | pending | pending |
| F-ODS | ResNet-50 | pending | pending | pending | pending |
| Dynamic learning | Convolutional | pending | pending | pending | pending |
| Dynamic learning | Pruned Vision Transformer | pending | pending | pending | pending |
| Dynamic learning | ResNet-50 | pending | pending | pending | pending |
| F-ODS and dynamic learning | Convolutional | pending | pending | pending | pending |
| F-ODS and dynamic learning | Pruned Vision Transformer | pending | pending | pending | pending |
| F-ODS and dynamic learning | ResNet-50 | pending | pending | pending | pending |

*Table 4. F-ODS and farm-knowledge-guided dynamic learning. Pending the rerun on NVIDIA GeForce RTX 4090(48GB). If the identity-safe unlabeled pool is empty, the dynamic-learning rows are reported as not run rather than filled.*

Also pending for this group: $(N^{\star},m^{\star})$, the five quality weights, $\lambda$, $\mu$, the gates $\tau_c$ and $\tau_q$, the refresh interval $R$, the number of admitted patches, and the wall-clock time of each row.

### 4.3 Qualitative comparison

Figure 1 will show one identity-disjoint test query and the gallery patch ranked first by each variant. The three panels are empty until the rerun exports them. No substitute image is used.

| Convolutional | Pruned Vision Transformer | ResNet-50 |
| --- | --- | --- |
| pending: query and top-ranked gallery patch | pending: query and top-ranked gallery patch | pending: query and top-ranked gallery patch |

*Figure 1. Qualitative comparison of the three variants. Pending export from the NVIDIA GeForce RTX 4090(48GB) rerun.*

## 5. Discussion

Once the cells are filled, the comparison is between trunks on the images as stored, and between the full training pool and a pool restricted by $Q$ and by the two gates. It is not a mask-versus-box comparison, because neither crop is built.

The herd is one group of 96 animals, the group described by Wang et al. (2025a). A rank on that group does not by itself transfer to another breed or another farm. The test is closed-set: every query identity is enrolled. An animal that is not in the gallery is outside the reported metrics. Checkpoint selection uses validation identities, so the test set is not used to pick the weights. Magnitude pruning reduces the transformer without asking whether a unit separates identities; that question is left open.

Wang et al. (2025a) remain the reference for what coat-pattern metric learning achieves on this herd under their practice. Agreement or disagreement with their figures will be interpretable only after our rerun, and only as a comparison of protocols, not as a claim on their numbers.

## 6. Conclusions

On the UAV-RGB herd of Wang et al. (2025a), three embeddings can be compared fairly when the coat patches are fixed in advance, the loss is shared and the split keeps identities apart. F-ODS and quality-gated self-training are specified on top of that comparison. The measurements themselves are pending the rerun on NVIDIA GeForce RTX 4090(48GB).

## CRediT authorship contribution statement

**Tianyi Ma:** Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Data curation, Writing – original draft, Writing – review and editing, Visualization, Project administration, Funding acquisition.

Role assignment is pending confirmation by the author.

## Declaration of competing interest

The author declares no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Acknowledgements

This work was supported by the Songshan Laboratory (No. 241110210200) and the Major Science and Technology Special Project of Henan Province (No. 221100210600).

Yaowu Wang and colleagues acquired the UAV imagery and released it. This study did not collect it.

## Data availability

The UAV-RGB imagery is the archive of Wang (2024), https://doi.org/10.5281/zenodo.13364663, CC BY 4.0, which is the UAV-RGB herd used by Wang et al. (2025a). Code for the protocol in this paper is at https://github.com/tianyi-star/2d_cattle under the Apache License 2.0. Trained weights and the filled tables are not yet deposited; their location is pending.

## References

Andrew, W., Gao, J., Mullan, S., Campbell, N., Dowsey, A.W., Burghardt, T., 2021. Visual identification of individual Holstein-Friesian cattle via deep metric learning. Computers and Electronics in Agriculture 185, 106133. https://doi.org/10.1016/j.compag.2021.106133

Deng, J., Guo, J., Xue, N., Zafeiriou, S., 2019. ArcFace: Additive angular margin loss for deep face recognition, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 4685–4694. https://doi.org/10.1109/CVPR.2019.00482

Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., Houlsby, N., 2021. An image is worth 16x16 words: Transformers for image recognition at scale, in: International Conference on Learning Representations. arXiv:2010.11929.

He, K., Zhang, X., Ren, S., Sun, J., 2016. Deep residual learning for image recognition, in: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 770–778. https://doi.org/10.1109/CVPR.2016.90

Wang, Y., 2024. Dataset for cattle growth monitoring in PLF using UAV-based LiDAR. Zenodo. https://doi.org/10.5281/zenodo.13364663

Wang, Y., Mücher, S., Wang, K., Wang, W., Kooistra, L., 2025a. Deep metric learning for individual cattle identification using coat patterns: Proposal for a best practice. Computers and Electronics in Agriculture 238, 110754. https://doi.org/10.1016/j.compag.2025.110754
