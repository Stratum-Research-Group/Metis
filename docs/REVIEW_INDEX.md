# Review index

**For first-time implementation, start with [the production build prompts](BUILD_PROMPTS.md).** The review files below record the reasoning behind those prompts.

The original `.txt` prompts are paired duplicates: each `v2` file repeats its earlier brief and adds a blanket LTHCM/Dianoia submodule requirement. Treat `v2` as the later draft, but use the reviews below before implementing it. These files review the supplied documents; they do not claim that the proposed components have been built or benchmarked.

| Prompt component | Monorepo Documentation |
| --- | --- |
| Paideia | [paideia/README.md](../paideia/README.md) & [REVIEW_AND_GAPS.md](../paideia/REVIEW_AND_GAPS.md) |
| Phylax | [phylax/README.md](../phylax/README.md) & [REVIEW_AND_GAPS.md](../phylax/REVIEW_AND_GAPS.md) |
| Kybernetes | [kybernetes/README.md](../kybernetes/README.md) & [REVIEW_AND_GAPS.md](../kybernetes/REVIEW_AND_GAPS.md) |
| Prosopon | [prosopon/README.md](../prosopon/README.md) & [REVIEW_AND_GAPS.md](../prosopon/REVIEW_AND_GAPS.md) |

The existing core components have detailed local documentation: [Stoma](../stoma/README.md), [Dianoia](../dianoia/README.md), [Dokime](../dokime/README.md), [Hestia](../hestia/README.md), [LTHCM](../LTHCM/README.md), [Kleroterion](../kleroterion/README.md), and [Krisis](../krisis/README.md). The overarching integration contract is [the stack integration plan](stack-integration.md). For the autonomous coding agent, Lochos multi-agent squads, and Pinax tool catalogue, see [Metis Sovereign Stack User Manual](MSS.md).


## Historical Reviews and Architecture Specifications (`docs/review/`)

| File | Topic / Scope |
| --- | --- |
| [mss-pyle-webrtc-passkey-spec.md](review/mss-pyle-webrtc-passkey-spec.md) | MSS Pyle: Modern WebRTC DataChannels & WebAuthn/Passkey Architecture Specification |
| [metis-ncm-anadrome-pyle-synthesis-roadmap-2026-10-18.md](review/metis-ncm-anadrome-pyle-synthesis-roadmap-2026-10-18.md) | NCM (Schedion, Symbolon, Anabathmos), Anadrome Walk Replay, and MSS Pyle Remote Access Synthesis & Roadmap |
| [mss-pyle-remote-access-devices-spec-c949cfb0.txt](review/mss-pyle-remote-access-devices-spec-c949cfb0.txt) | MSS Pyle Remote Access, Device Identity & Management Specification (Historical Base) |

| [metis-anadrome-walk-replay-experiment-spec-aa69f396.txt](review/metis-anadrome-walk-replay-experiment-spec-aa69f396.txt) | Metis Anadrome Walk-Trace Replay Policy Optimization Specification |
| [metis-ncm-schedion-integration-spec-2026-10-18-a3f41c7d-2aa95bb2.txt](review/metis-ncm-schedion-integration-spec-2026-10-18-a3f41c7d-2aa95bb2.txt) | Metis x NCM Schedion Coarse-to-Fine Candidate Generation Specification |
| [lthcm-ncm-symbolon-integration-spec-2026-10-18-c51d8a62-8951392a.txt](review/lthcm-ncm-symbolon-integration-spec-2026-10-18-c51d8a62-8951392a.txt) | LTHCM x NCM Symbolon Discrete Code Auxiliary Indexing Specification |
| [dioptra-ncm-anabathmos-integration-spec-2026-10-18-b7e29d04-13b8d9f4.txt](review/dioptra-ncm-anabathmos-integration-spec-2026-10-18-b7e29d04-13b8d9f4.txt) | Dioptra x NCM Anabathmos Hierarchical Interpretability Specification |
| [01-embedder-identity-lthcm-metis-6cd9ff83.txt](review/01-embedder-identity-lthcm-metis-6cd9ff83.txt) | Exact Embedder Identity & Representation Compatibility Architecture |


| [02-evaluation-gate-integrity-dokime-krisis-87c5fa3f.txt](review/02-evaluation-gate-integrity-dokime-krisis-87c5fa3f.txt) | Evaluation-Gate Integrity & Adversarial Fixture Discipline |
| [03-recoverable-model-swaps-metis-a2853c8a.txt](review/03-recoverable-model-swaps-metis-a2853c8a.txt) | Recoverable Model-Swap Lifecycle for Hestia |
| [metis-grpo-walk-policy-spec-e6633b6a.txt](review/metis-grpo-walk-policy-spec-e6633b6a.txt) | GRPO Walk Policy Optimization for Dianoia Deliberative Walks |
| [metis-repo-wide-round4-369939e-findings-7ced56a6.txt](review/metis-repo-wide-round4-369939e-findings-7ced56a6.txt) | Round 4 Repo-wide Review (v6) Findings & Verification |
| [metis-openrig-lochos-integration-spec-a41d7c29-fb788990.txt](review/metis-openrig-lochos-integration-spec-a41d7c29-fb788990.txt) | Lochos Multi-Agent Squad & Durable Work Queue Specification |
| [metis-pinax-tool-catalogue-spec-83941186.txt](review/metis-pinax-tool-catalogue-spec-83941186.txt) | Pinax System Tool Catalogue & FTS5 Discovery Specification |
| [metis-dashboard-implementation-spec-ae93d36c.txt](review/metis-dashboard-implementation-spec-ae93d36c.txt) | Prosopon Admin & Workbench Architecture Specification |
| [metis-repo-wide-round1-fixes-6752222c.txt](review/metis-repo-wide-round1-fixes-6752222c.txt) | Round 1 Review Findings & Remediations |
| [metis-repo-wide-round2-45f0e72-fixes-be5f1a23.txt](review/metis-repo-wide-round2-45f0e72-fixes-be5f1a23.txt) | Round 2 Review Findings & Remediations |
| [metis-round2-bcac410-fixes-002bd2aa.txt](review/metis-round2-bcac410-fixes-002bd2aa.txt) | Round 2 Delta Fixes |
| [metis-review-aacb3e58-71bc604c.txt](review/metis-review-aacb3e58-71bc604c.txt) | Initial Foundation Review |
| [anadrome-frozen-live-walk-milestone3-report.md](review/anadrome-frozen-live-walk-milestone3-report.md) | Anadrome Milestone 3.2: Frozen Live Walk Test & Policy Decision Gate Report |
| [schedion-stage3-controlled-comparison-report.md](review/schedion-stage3-controlled-comparison-report.md) | Schedion Stage 3: Controlled Comparison & Krisis Gate Report (ARCHIVE) |
| [symbolon-candidate-fusion-e2-report.md](review/symbolon-candidate-fusion-e2-report.md) | Symbolon E2: Candidate Capture & Fusion Study Report (ARCHIVE) |
| [symbolon-reconstruction-fidelity-e1-report.md](review/symbolon-reconstruction-fidelity-e1-report.md) | Symbolon E1: Multiscale Discrete Reconstruction Fidelity Diagnostic Report |
| [ncm-hardware-validation-rtx5060ti-report.md](review/ncm-hardware-validation-rtx5060ti-report.md) | NCM Hardware & Memory Validation on RTX 5060 Ti 16GB Report |
| [metis-substrate-review-d2b97509-f9bb42d2.txt](review/metis-substrate-review-d2b97509-f9bb42d2.txt) | Agent Substrate (gVisor/Cloud Hypervisor) for Metis: Source Review & Execution Boundary Analysis |
| [metis-billion-context-review-cda8c837-4e379b25.txt](review/metis-billion-context-review-cda8c837-4e379b25.txt) | Billion-Context (acp-kernel) for Metis: Source Review & Context Management Analysis |
| [harness2-repo-review-1cfe254-38994399.txt](review/harness2-repo-review-1cfe254-38994399.txt) | Google Research Harness2 (Recursive Agent Harnessing): Source Review & Guardrail Safety Analysis |
| [LTHCM-recovery-application-plan-356cbcd7.txt](review/LTHCM-recovery-application-plan-356cbcd7.txt) | LTHCM Topology & Sheaf Recovery Plan |

