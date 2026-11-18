# Citation spot-check

Mark each claim: does the quoted passage actually support it?


## What problem with existing self-supervised methods does the paper say it addresses?
supported=True  quotes verified 1/1

- **[SOURCE]** In this work, we explore how to improve the semantic level of self-supervised representations without using extra prior knowledge encoded through image transformations. To that end, we introduce a joint-embedding predictive architecture [48] for images (I-JEPA).
    - [ ] verified §1 Introduction (p.2): “In this work, we explore how to improve the semantic level of self-supervised representations without using extra prior knowledge encoded through image transformations. To that end, we introduce a joint-embedding predictive architecture [48] for images (I-JEPA).”
      > resulting representations are typically of a lower semantic level and underperform invariance-based pretraining in offthe-shelf evaluations (e.g., linear-probing) and in transfer settings with limited supervision for semantic classification tasks [4]. Consequently, a more involved adaptation mechanism (e.g., end-to-end fine-tuning) is required to reap the full advantage of these methods. In this work, we explore how to improve the semantic level of self-supervised representations without using e

## What is the predictor and what does it take as input?
supported=True  quotes verified 1/1

- **[SOURCE]** The predictor gϕ(·, ·) takes as input the output of the context encoder sx and a mask token for each patch we wish to predict, {mj}j∈Bi, and outputs a patch-level prediction ˆsy(i) = {ˆsyj}j∈Bi = gϕ(sx, {mj}j∈Bi).
    - [ ] verified §3 Method (p.4): “the predictor gϕ(·, ·) takes as input the output of the context encoder sx and a mask token for each patch we wish to predict, {mj}j∈Bi, and outputs a patch-level prediction ˆsy(i) = {ˆsyj}j∈Bi = gϕ(sx, {mj}j∈Bi).”
      > patch-level representation. Typically, we set M equal to 4, and sample the blocks with a random aspect ratio in the range (0.75, 1.5) and random scale in the range (0.15, 0.2). Note that the target blocks are obtained by masking the output of the target-encoder, not the input. This distinction is crucial to ensure target representations of a high semantic level; see, e.g., [8]. Context. Recall, the goal behind I-JEPA is to predict the target block representations from a single context block. To 

## What does the paper say about how the target encoder is updated?
supported=True  quotes verified 1/1

- **[SOURCE]** The target representations correspond to the outputs of the target-encoder, the weights of which are updated at each iteration via an exponential moving average of the context encoder weights.
    - [ ] verified §3 Method (Figure 3, p.3): “the weights of which are updated at each iteration via an exponential moving average of the context encoder weights.”
      > Figure 3. I-JEPA. The Image-based Joint-Embedding Predictive Architecture uses a single context block to predict the representations of various target blocks originating from the same image. The context encoder is a Vision Transformer (ViT), which only processes the visible context patches. The predictor is a narrow ViT that takes the context encoder output and, conditioned on positional tokens (shown in color), predicts the representations of a target block at a specific location. The target re

## What is the training objective?
supported=False  quotes verified 1/1

- **[INFERENCE]** The training objective is to minimize the loss function ∥hω(ˆx, sx)−ϵ∥, where the decoder network hω aims to reconstruct an image x from the representation vector sx and a noisy version of that image ˆx.
    - [ ] verified §E RCDM Visualizations (p.16): “The decoder objective is to minimize the loss function ∥hω(ˆx, sx)−ϵ∥.”
      > To visualize the representations of a pretrained neural network in pixel space, we use the RCDM framework [13]. The RCDM framework trains a decoder network hω, comprising a generative diffusion model, to reconstruct an image x from the representation vector of that image sx and a noisy version of that image ˆx := x + ϵ, where ϵ is an additive noise vector. Concretely, the decoder objective is to minimize the loss function ∥hω(ˆx, sx)−ϵ∥. We train each RCDM network for 300,000 iterations using th

## How are context and target blocks chosen?
supported=True  quotes verified 2/2

- **[SOURCE]** Context and target blocks are chosen as follows: context blocks are sampled with a random scale in the range (0.85, 1.0) and unit aspect ratio. Target blocks are sampled with a random aspect ratio in the range (0.75, 1.5) and random scale in the range (0.15, 0.2).
    - [ ] verified §3 Method (Figure 4, p.4): “Given an image, we randomly sample 4 target blocks with scale in the range (0.15, 0.2) and aspect ratio in the range (0.75, 1.5). Next, we randomly sample a context block with scale in the range (0.85, 1.0) and remove any overlapping target blocks.”
      > Figure 4. Examples of our context and target-masking strategy. Given an image, we randomly sample 4 target blocks with scale in the range (0.15, 0.2) and aspect ratio in the range (0.75, 1.5). Next, we randomly sample a context block with scale in the range (0.85, 1.0) and remove any overlapping target blocks. Under this strategy, the target-blocks are relatively semantic, and the contextblock is informative, yet sparse (efficient to process).
    - [ ] verified §3 Method (p.4): “To obtain the context in I-JEPA, we first sample a single block x from the image with a random scale in the range (0.85, 1.0) and unit aspect ratio.”
      > patch-level representation. Typically, we set M equal to 4, and sample the blocks with a random aspect ratio in the range (0.75, 1.5) and random scale in the range (0.15, 0.2). Note that the target blocks are obtained by masking the output of the target-encoder, not the input. This distinction is crucial to ensure target representations of a high semantic level; see, e.g., [8]. Context. Recall, the goal behind I-JEPA is to predict the target block representations from a single context block. To 

## What does the paper claim about predicting in representation space versus pixel space?
supported=True  quotes verified 2/2

- **[SOURCE]** predicting in representation space leads to a significant degradation in the linear probing performance compared to pixel-space prediction.
    - [ ] verified §9 Ablations (p.7): “predicting in pixel-space leads to a significant degradation in the linear probing performance.”
      > Predicting in representation space. Table 7 compares low-shot performance on 1% ImageNet-1K using a linear probe when the loss is computed in pixel-space versus representation space. We conjecture that a crucial component of I-JEPA is that the loss is computed entirely in representation space, thereby giving the target encoder the ability to produce abstract prediction targets, for which irrelevant pixel-level details are eliminated. From Table 7, it is clear that predicting in pixel-space leads
- **[SOURCE]** I-JEPA learns strong off-the-shelf representations without the use of hand-crafted view augmentations, and outperforms pixel-reconstruction methods such as MAE on ImageNet-1K linear probing.
    - [ ] verified §1 Introduction (p.2): “I-JEPA outperforms pixel-reconstruction methods such as MAE [36] on ImageNet-1K linear probing, semi-supervised 1% ImageNet-1K, and semantic transfer tasks.”
      > resulting representations are typically of a lower semantic level and underperform invariance-based pretraining in offthe-shelf evaluations (e.g., linear-probing) and in transfer settings with limited supervision for semantic classification tasks [4]. Consequently, a more involved adaptation mechanism (e.g., end-to-end fine-tuning) is required to reap the full advantage of these methods. In this work, we explore how to improve the semantic level of self-supervised representations without using e

## What ablations does the paper report?
supported=True  quotes verified 2/2

- **[SOURCE]** The paper reports ablations on weight decay strategies, multiblock masking strategies, and target masking strategies.
    - [ ] verified §C Additional Ablations (p.15): “In Table 13, we evaluate the impact of weight-decay during pretraining.”
      > Weight decay. In Table 13, we evaluate the impact of weight-decay during pretraining. We explore two weight decay strategies: linearly increase the weight-decay from 0.04 to 0.4 or use a fix weight-decay of 0.05. Using a smaller weight decay during pretraining improves the downstream performance on ImageNet-1% when fine-tuning. However, this also leads to a degradation of performance in linear evaluation. In the main paper, we use the first weight decay strategy as it improves the performances i
    - [ ] verified §C Additional Ablations (p.14): “This section follows the same experimental protocol as Section 9. We report the result of a linear probe with a frozen backbone, trained on the low-shot 1% ImageNet-1K benchmark.”
      > This section follows the same experimental protocol as Section 9. We report the result of a linear probe with a frozen backbone, trained on the low-shot 1% ImageNet-1K benchmark. Multiblock masking strategy. We present an extended ablation of the multiblock masking strategy where we change the targets block scale (Table 8), the context scale (Table 9) and the number of target blocks (Table 10). We train a ViT-B/16 for 300 epochs using I-JEPA with various multi-block settings and compare performa

## Does the paper say anything about video?
supported=False  quotes verified 0/0


## What GPU does the paper train on and for how many hours?
supported=False  quotes verified 0/0


## Who is the senior author's favorite color?
supported=False  quotes verified 0/0
