# ASOS: current open models and medical model guide

**Research date:** 30 September 2026  
**Project:** [synaptek-labs/synaptek-project](https://gitlab.com/synaptek-labs/synaptek-project)  
**Purpose:** Select models to evaluate and integrate; this document does not install models or establish clinical readiness.

## Recommended starting point

For ASOS, evaluate **Qwen3.8-27B as the general assistant**, **MedGemma 1.5 4B as the medical image/text specialist**, and **Qwen3 embeddings plus a reranker for grounded document retrieval**. On smaller hardware, start with Qwen3.5-9B. For a permissively licensed medical text baseline, include MediPhi; with substantially more memory, evaluate Apertus-70B-MeditronFO.

These are engineering recommendations based on the official model cards linked below, not measured winners on ASOS. Hardware, concurrency, intended medical tasks, languages and deployment budget have not been supplied. The newest giant model is not automatically the best choice for this project.

The current-source shortlist includes Qwen3.8, Gemma 4, DeepSeek V4.1, GLM-5.3 and Kimi K3. Older specialist models remain useful where their role, license or resource requirements fit better. “Current” means official weights/model documentation were available when checked; it does not mean every row was released this month, or that this is an exhaustive market inventory. Website update dates are not treated as release dates.

## 1. Open source versus open weights

The tables distinguish **Apache-2.0/MIT licensed weights** from **weights under custom terms**. Downloadable weights alone do not establish that training data, recipes and the entire development process are open. Do not describe every model here as unrestricted open source.

MedGemma, MedSigLIP and MedASR use Google's **Health AI Developer Foundations (HAI-DEF)** terms. Gemma 4's Apache-2.0 license does not carry over to those medical models. GLM-5.3 and Kimi K3 also have checkpoint-specific custom licenses. Preserve the exact license and model revision during procurement and deployment; check derivative and quantized checkpoints separately.

## 2. General, reasoning and coding models

| Model and official checkpoint | Verified characteristics | Suggested ASOS role | License and decision |
|---|---|---|---|
| **[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)** — `Qwen/Qwen3.8-27B` | Dense language model with native image/video input; 27B language parameters, with additional vision components. Native 262,144-token context, extendable with the documented configuration. | First general-purpose candidate for planning, grounded research synthesis, document understanding and code assistance. Start with a much smaller operational context limit. | **Apache-2.0. Recommended first evaluation.** The card documents Transformers, vLLM and SGLang paths; pin a version supporting this architecture. |
| **[Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B)** — `Qwen/Qwen3.5-9B` | Smaller image/text model from the preceding generation. | Low-memory assistant, routing and simple extraction; useful comparison against the larger default. | **Apache-2.0. Small-hardware candidate.** Check tool behavior and medical performance independently. |
| **[Gemma 4 31B IT](https://huggingface.co/google/gemma-4-31B-it)** — `google/gemma-4-31B-it` | Instruction-tuned image/text model in the 31B class. | Alternative general assistant and visual document model to compare with Qwen. | **Apache-2.0 for this checkpoint.** General model, not a medical specialist. |
| **[Qwen3-Coder-Next](https://huggingface.co/Qwen/Qwen3-Coder-Next)** — `Qwen/Qwen3-Coder-Next` | 80B total / 3B active MoE parameters; 262,144 native context; non-thinking coding model. | Dedicated coding and tool-workflow service if it beats the general model on ASOS tasks. | **Apache-2.0. Optional larger deployment.** Allocate memory for total weights, not 3B active parameters. |
| **[DeepSeek V4.1 Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)** — `deepseek-ai/DeepSeek-V4.1-Flash` | Image/text model; 552B backbone plus 196B Engram conditional memory; listed complete artifact approximately 763B parameters. Activated parameters differ between prefill and decode. | Cluster-hosted reasoning/research candidate after smaller models are benchmarked. | **MIT. Cluster only for the proposed shortlist.** Follow its reference encoding and serving recipe; a generic chat-template assumption is insufficient. |
| **[GLM-5.3](https://huggingface.co/zai-org/GLM-5.3)** — `zai-org/GLM-5.3` | Large text-generation MoE; model repository reports approximately 753B parameters and includes FP8 tensors. | Optional cluster-hosted coding/reasoning comparison. | **Custom GLM-5.3 license.** Do not infer MIT terms from earlier GLM releases. |
| **[Kimi K3](https://huggingface.co/moonshotai/Kimi-K3)** — `moonshotai/Kimi-K3` | Native multimodal model; 2.8T total / 104B active parameters, approximately 1M context, native MXFP4 weights. | Large-cluster research and complex tool-workflow evaluation. | **Custom Kimi K3 license.** Very large infrastructure commitment; not an initial local deployment. |

Model-card benchmark scores are publisher results under particular prompts, tools and budgets. They are not comparable ASOS end-to-end measurements. This guide deliberately does not turn those scores into an unsupported ranking.

## 3. Medical and biomedical models

| Model and official checkpoint | Useful capability | Recommended use in ASOS | Important boundary |
|---|---|---|---|
| **[MedGemma 1.5 4B IT](https://huggingface.co/google/medgemma-1.5-4b-it)** — `google/medgemma-1.5-4b-it` | Medical text and image comprehension; updated support for medical documents, longitudinal imaging, volume representations and pathology patches. | First medical multimodal specialist to evaluate. Send bounded cases through a dedicated image/text adapter. | **HAI-DEF.** This 1.5 release is the 4B variant; do not invent a “1.5 27B” checkpoint. Its card says multi-turn use was not optimized/evaluated. It is not a validated autonomous diagnostic service. |
| **[MedGemma 27B text IT](https://huggingface.co/google/medgemma-27b-text-it)** — `google/medgemma-27b-text-it` | Larger medical text model. | Compare on evidence-grounded medical question answering and report synthesis when memory permits. | **HAI-DEF.** This exact checkpoint is text-only; do not send images to it or assume a newer version from its name. |
| **[MediPhi](https://huggingface.co/microsoft/MediPhi)** — `microsoft/MediPhi` | Approximately 3.8B parameters; clinical NLP based on Phi-3.5-mini. | Small medical text baseline for entity extraction, note structuring and summarization. | **MIT.** A 2025 specialist, not a newly released frontier model. Intended research use is English clinical NLP; separately validate other languages and settings. |
| **[Apertus-70B-MeditronFO](https://huggingface.co/EPFLiGHT/Apertus-70B-MeditronFO)** — `EPFLiGHT/Apertus-70B-MeditronFO` | 70B medical text specialist from the Fully Open Meditron effort, with published data/training provenance. | Larger medical-text research candidate where auditability matters and hardware permits. | **Apache-2.0.** The [May 2026 paper](https://arxiv.org/abs/2605.16215) describes the auditable pipeline. Published medical benchmark improvements do not establish local clinical effectiveness. |
| **[MedSigLIP](https://huggingface.co/google/medsiglip-448)** — `google/medsiglip-448` | Medical image/text embeddings for retrieval and downstream classification. | Index medical images or retrieve similar studies; evaluate a trained classifier on relevant labelled data. | **HAI-DEF.** An encoder, not a conversational model or segmentation system. Use a separate embedding collection and service. |
| **[MedASR](https://huggingface.co/google/medasr)** — `google/medasr` | Medical speech transcription. | Optional dictation service feeding a separately evaluated note-structuring workflow. | **HAI-DEF.** English training with substantial US-speaker representation; assess local accents, microphone quality, drug names, numbers and negation. Outputs need verification. |

For ASOS's scientific work, select a model by the actual job. Medical document understanding, microscopy segmentation, transcription and numerical statistics are different capabilities. A medical LLM does not automatically perform all of them.

### Microscopy and cell analysis

**[Cellpose 4 / Cellpose-SAM](https://github.com/MouseLand/cellpose)** is a relevant segmentation candidate for cell and nucleus workflows. Its repository documents June 2026 additions including `cpsam_v2` and DINOv3-based variants. The [software license](https://github.com/MouseLand/cellpose/blob/main/LICENSE) is BSD-3-Clause, but upstream explicitly identifies CC-BY-NC training data. Review the chosen weights, backbone and data terms independently before commercial use; the software license alone does not resolve those questions.

Use segmentation masks to compute counts, areas and distributions through deterministic measurement code. Validate on ASOS's own preparation, staining, microscope and magnification. This is a proposed replacement path for demonstration cell-counting behavior, not evidence of an accurate microclot detector. No model in this guide has been validated for that ASOS-specific target.

## 4. Retrieval models and evidence grounding

| Candidate | Relevant facts | ASOS integration decision |
|---|---|---|
| **[Qwen3-Embedding-4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B)** | Apache-2.0; 32K input context; output dimensions configurable from 32 to 2560. | Evaluate on papers, protocols and local documents. Choose and freeze dimensions, normalization, query instructions and chunking together. |
| **[Qwen3-Reranker-4B](https://huggingface.co/Qwen/Qwen3-Reranker-4B)** | Apache-2.0; dedicated relevance-ranking model. | Rerank an initial candidate set before evidence is sent to the answering model. Measure the additional latency against retrieval improvement. |
| **Smaller Qwen3 retrieval variants** | The [official family table](https://huggingface.co/Qwen/Qwen3-Embedding-4B) includes 0.6B embedding and reranking variants; the 0.6B embedding model supports 1024 dimensions. | Lower-resource comparison when the 4B pair consumes too much memory. Confirm each chosen artifact's revision and terms before deployment. |

ASOS's reviewed registry already lists BGE-M3, and its database schema assumes 1024-dimensional vectors. Keep that configuration as a baseline after making real embedding execution work. Even if a replacement emits 1024 dimensions, its vectors are in a different semantic space: **re-embed the entire corpus and do not mix old and new vectors**. With another dimension, migrate the database/vector collection and rebuild indexes as well.

Retrieve from a versioned corpus with document IDs, passage spans, publication dates and access controls. Answer citations must resolve to retrieved evidence. Neither a medical specialist nor a larger context window replaces that requirement.

## 5. Hardware planning

These are **planning estimates, not tested requirements or purchasing guarantees**. Quantization availability and runtime support must be verified for the exact architecture. Start with one request and a modest context, such as 8K, then measure realistic document and image loads.

Approximate raw dense weight storage in decimal GB is `parameters in billions × bytes per parameter`: BF16/FP16 about 2 bytes; idealized 4-bit about 0.5 bytes. This excludes quantization metadata, unquantized tensors, vision components, caches, activations and serving overhead. Disk size, system RAM and GPU memory are separate budgets.

| Deployment size | Practical experiment to plan | Memory qualification |
|---|---|---|
| **8–16 GB GPU** | A 3.8–4B medical model, or a quantized 9B general model, loaded separately. | Four-billion-parameter BF16 weights alone are roughly 8 GB. An 8 GB GPU therefore needs a smaller/quantized setup; image and context overhead can be significant. |
| **24 GB GPU** | Start with quantized Qwen3.5-9B and schedule a medical specialist separately. Experiment with quantized Qwen3.8-27B only after runtime support is confirmed. | A 27B model has approximately 13.5 GB idealized 4-bit weights before extras. Fit and useful concurrency are not guaranteed; avoid assuming the entire stack remains resident. |
| **48 GB GPU** | Quantized 27B/31B general model with more operating room; load or separately host the specialist/retriever as measured. | BF16 27B weights alone are around 54 GB, so full BF16 does not fit this budget. |
| **80 GB GPU** | Evaluate BF16 27B-class models at bounded context, or quantized 70B/80B-class candidates. | 70B/80B at idealized 4-bit means about 35/40 GB weights. Long context, concurrent users and runtime overhead still constrain capacity. |
| **Multiple large GPUs** | Higher-precision 70B/80B serving or the frontier cluster candidates. | 70B BF16 is roughly 140 GB before overhead. DeepSeek V4.1, GLM-5.3 and especially Kimi K3 require a separate model-specific capacity and interconnect plan. |

MoE active parameters primarily describe computation per token; **resident weight planning must account for total parameters and any conditional memory**. CPU offload may help fit a model but changes latency and throughput. ASOS's distributed node inventory does not itself provide working tensor/pipeline parallelism across those nodes.

## 6. Changes ASOS needs before these models are usable

This assessment uses the locally retained source archive at **`860841260ea55c8236f363f58d843d8539ab8906`**, matching [ASOS_REVIEW.md](ASOS_REVIEW.md). The remote branch was not refreshed for this model guide. These are findings about that snapshot, not claims about subsequent commits.

| Priority | Verified project gap | Required improvement and acceptance evidence |
|---|---|---|
| **P0** | Model loading is bookkeeping rather than a demonstrated inference-worker lifecycle. | Start a real worker, download/pin its artifact, register its actual endpoint, and mark ready only after successful inference. Prove an ASOS request reaches it and returns its model identity and usage. |
| **P0** | Scientific agents/tools contain predetermined or synthetic results. | Replace demonstrations with genuine model/tool calls. Statistical outputs must come from tested numerical implementations. Demonstrate changed input changes the result, and missing data produces failure rather than invented findings. |
| **P0** | The prior review identifies unrestricted Python execution and incomplete authorization. | Isolate model-requested tools, enforce operation-level permissions and resource limits, and test cancellation. Complete these controls before giving a new coding/agent model access to instruments, private records or arbitrary execution. |
| **P1** | `InferenceRequest.messages` is `List[Dict[str, str]]`; the normal path is text-oriented and constructs `stream=False`. | Add typed image/text parts, tool-call messages, tool results and validated generation options. Preserve tool calls in responses and implement/test streaming separately. Verify an image reaches the vision processor unchanged. |
| **P1** | Registry metadata is insufficient for new architectures; the Scout entry records 17B in a field labelled total parameters. | Separate total/active parameters, architecture, modalities, artifact revision, quantization format and backend. Include processor/tokenizer versions, license and measured memory. Remove unsafe generic size assumptions. |
| **P1** | Routing uses older model tiers and generic complexity rules. | Route by capability, language, privacy, measured task quality and available capacity. Provide explicit model overrides and a fail-closed policy for unsupported modalities. |
| **P1** | Embedding execution and fixed vector dimensions need attention. | Implement real embeddings, version indexes, rebuild on model changes and test retrieval against labelled queries. Keep image embeddings separate from text embedding spaces. |
| **P1** | Serving formats and context costs are treated too generically. | Validate exact runtime/kernel/quantization combinations. Perform a load test with the actual context, image count and concurrency; do not advertise the card's maximum context as available capacity. |
| **P1** | Medical outputs lack task-specific quality evidence. | Store provenance and review state, evaluate locally representative data, and prevent preliminary outputs from silently becoming finalized clinical records. |

Source anchors: [model registry](https://gitlab.com/synaptek-labs/synaptek-project/-/blob/860841260ea55c8236f363f58d843d8539ab8906/asos/pool/model_registry.py), [request/response and forwarding](https://gitlab.com/synaptek-labs/synaptek-project/-/blob/860841260ea55c8236f363f58d843d8539ab8906/asos/pool/load_balancer.py), [model selector](https://gitlab.com/synaptek-labs/synaptek-project/-/blob/860841260ea55c8236f363f58d843d8539ab8906/asos/router/model_selector.py), and the individual evidence links in [the repository review](ASOS_REVIEW.md).

### Proposed service arrangement

Use ASOS as the authenticated router and workflow coordinator, with independent services for:

1. General chat/reasoning and optional dedicated coding.
2. Medical text/image processing with the model's own processor and template.
3. Text embeddings and reranking.
4. Medical image embeddings, segmentation and transcription where required.
5. Deterministic scientific computation and report rendering.

An OpenAI-compatible HTTP interface can simplify text-model transport, but it does not prove tool, multimodal or reasoning-format compatibility. Follow each model's official serving recipe. Use specialized endpoints for embeddings, masks and audio. Persist a manifest for each service containing checkpoint revision, weight checksums, license, runtime/container version, processor, quantization, context limits and evaluation results.

## 7. Evaluation before selection

Run every comparison through ASOS's actual request path, with the same frozen inputs and resource budgets. Establish thresholds before inspecting results; record failures and timeouts as failures rather than excluding them.

| Workload | Evidence to collect |
|---|---|
| General research assistant | Task completion, verified citation support, unsupported statements, valid structured output, tool-call correctness and refusal on missing evidence. |
| Clinical text | Expert-labelled entity/field extraction F1, omissions and additions in summaries, medication/dose/unit/negation errors, and performance by document type and language. |
| Medical image understanding | Blinded expert assessment on the relevant modality; sensitivity/specificity only for a defined labelled task, with patient-level separation and confidence intervals. |
| Microscopy segmentation | Instance segmentation quality, count error, boundary errors and robustness across staining, acquisition devices and blank/low-quality images. |
| Retrieval | Recall@k and nDCG on biomedical and local-document queries; citation correctness after generation; access-control leakage tests. |
| Transcription | Word error rate plus medical entity, number, dose and negation errors; subgroup results by accent and recording conditions. |
| Operations | Cold start, time to first token, p50/p95 latency, throughput, peak VRAM/RAM, queue behavior, cancellation and recovery after worker failure. |

Use public benchmarks as supplementary evidence, then held-out data representative of the intended application. Keep patient-related data access controlled and separate training from evaluation. Compare the medical specialist against the general model with the same retrieval evidence: specialization should earn its place through measured benefit.

No fresh model inference, GPU capacity test, benchmark or clinical validation was performed for this document. No model is designated production-ready here.

## 8. Implementation order

1. **Make one general model work end to end.** Choose the small or balanced candidate according to hardware; prove real serving, failure propagation and provenance.
2. **Add grounded retrieval.** Compare the existing configured embedding baseline with the Qwen3 options, rebuild indexes and validate citations.
3. **Add one medical specialist.** Use MedGemma 1.5 4B for image/text tasks, or MediPhi for an initial small permissively licensed text workflow. Evaluate before expanding scope.
4. **Evaluate larger alternatives.** Compare MedGemma 27B, MeditronFO and a dedicated coding model only where the small stack has a measured weakness.
5. **Add task-specific vision/audio services.** Treat segmentation and dictation as independent products with their own data, preprocessing and acceptance tests.
6. **Consider giant frontier models last.** Require a demonstrated quality benefit that justifies the infrastructure and license obligations.

The concrete first experiment is a small, source-grounded ASOS workflow with a real worker and an evaluated medical specialist. Replacing model names in the registry alone will not deliver it.
