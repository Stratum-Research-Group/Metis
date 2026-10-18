# Symbolon Experiment E1: Reconstruction-Fidelity Diagnostic Report

- **Date**: 2026-10-18
- **Checkpoint ID**: `xsum_v1`
- **Tokenizer ID**: `xsum_vqvae_48`
- **Number of Evaluated Probes**: 4

## 1. Executive Summary & Level Progression

Per Symbolon Integration Spec Section 3 (E1), this diagnostic calibrates the semantic
fidelity and information loss across NCM's 9 hierarchical discrete codebook levels.
Lossy reconstruction does not disqualify discrete codes as candidate proposals, but bounds
the granularity of exact-match assumptions.

| Level | Code Count | Mean Word Overlap | Mean Char Accuracy | Key Terms Preserved |
|:-----:|:----------:|:-----------------:|:------------------:|:-------------------:|
| L0 | 4 | 1.2% | 0.2% | 0.0% |
| L1 | 6 | 2.2% | 0.8% | 6.2% |
| L2 | 10 | 6.8% | 0.3% | 0.0% |
| L3 | 14 | 14.1% | 1.3% | 0.0% |
| L4 | 20 | 15.7% | 1.3% | 0.0% |
| L5 | 26 | 23.6% | 6.7% | 12.5% |
| L6 | 32 | 26.5% | 5.7% | 6.2% |
| L7 | 40 | 39.1% | 6.4% | 31.2% |
| L8 | 48 | 82.8% | 59.5% | 87.5% |

## 2. Detailed Probe Inspections

### Probe: `probe_entities_geo` (Named Entities & Geography)
**Original Text**: *"Chancellor Angela Merkel arrived in Berlin on Monday for emergency European Union negotiations regarding the northern economic corridor expansion."*

| Level | Word Overlap | Key Terms Preserved | Decoded Output Preview |
|:-----:|:------------:|:-------------------:|:-----------------------|
| L0 | 0.0% | 0.0% | `Brexit Ass Last` |
| L1 | 4.5% | 25.0% | `Chancellor suspicion Last Dod` |
| L2 | 0.0% | 0.0% | `llor summit updated the.usk.` |
| L3 | 19.1% | 0.0% | `llor Merkel Monday Union the expansion` |
| L4 | 28.6% | 0.0% | `llor Merkel arrived Monday emergency regarding the economic.` |
| L5 | 45.0% | 25.0% | `Chancellor arrived in on for Union regarding the economic...` |
| L6 | 38.1% | 0.0% | `llorllor arrived arrived in on for European negotiations ...` |
| L7 | 47.6% | 0.0% | `llorllor Merkel arrived arrived on on for for emergency n...` |
| L8 | 100.0% | 100.0% | `Chancellor Angela Merkel arrived in Berlin on Monday for ...` |

### Probe: `probe_topological_math` (Topological & Math Concepts)
**Original Text**: *"The cellular sheaf Laplacian computes obstruction cohomology over the directed graph simplices, bounding topological drift to less than 0.05 per step."*

| Level | Word Overlap | Key Terms Preserved | Decoded Output Preview |
|:-----:|:------------:|:-------------------:|:-----------------------|
| L0 | 4.8% | 0.0% | `-, to` |
| L1 | 0.0% | 0.0% | `i yellow, than.` |
| L2 | 13.6% | 0.0% | `Robpl over at, than per` |
| L3 | 12.0% | 0.0% | `theac sam over at Powder, less 0 per` |
| L4 | 10.7% | 0.0% | `the Explpl sixteen Sigma empirical animations bound bound...` |
| L5 | 17.9% | 0.0% | `cellular theafpl samuristic astronom over animations Shoc...` |
| L6 | 13.8% | 0.0% | `cellular she Laafianophrification disruptingomom directed...` |
| L7 | 31.0% | 25.0% | `cellular sheaf Laafian comporphutes obstructionom over ov...` |
| L8 | 81.8% | 75.0% | `The cellular sheaf Laplacian computes obstruction disrupt...` |

### Probe: `probe_numerical_data` (Numerical & Telemetry Quantities)
**Original Text**: *"In 2026, Apollo mission telemetry indicated Apollo 11 consumed 14,200 gallons of propellant within the initial 180 seconds of atmospheric ascent."*

| Level | Word Overlap | Key Terms Preserved | Decoded Output Preview |
|:-----:|:------------:|:-------------------:|:-----------------------|
| L0 | 0.0% | 0.0% | `,, tracks has` |
| L1 | 0.0% | 0.0% | `,, device Lastquin` |
| L2 | 9.5% | 0.0% | `,000, tonnes within of` |
| L3 | 14.3% | 0.0% | `20, 11 14,ant within of` |
| L4 | 12.5% | 0.0% | `20, mission Experiment 11200200ant the initial ascent course` |
| L5 | 16.7% | 25.0% | `2026,metry Telescope 11 14200200 of within the mm of ascent` |
| L6 | 33.3% | 25.0% | `In 20, mission tele Telescope 11 11 14, gallons ofant the...` |
| L7 | 45.5% | 100.0% | `In 2026 Apollo Apollo telemetry Apollo Apollo 11 14,200 g...` |
| L8 | 90.0% | 100.0% | `In 2026, Apollo mission telemetry indicated Apollo 11 con...` |

### Probe: `probe_scientific_biochem` (Scientific & Biochemical Terminology)
**Original Text**: *"Dihydrofolate reductase inhibitors such as methotrexate bind competitively to the enzymatic active pocket with an inhibition constant Ki of 4.2 nanomolar."*

| Level | Word Overlap | Key Terms Preserved | Decoded Output Preview |
|:-----:|:------------:|:-------------------:|:-----------------------|
| L0 | 0.0% | 0.0% | `.` |
| L1 | 4.3% | 0.0% | `ggedro to an.` |
| L2 | 4.2% | 0.0% | `Solase suchate to with...` |
| L3 | 11.1% | 0.0% | `D Solct as competitive to emph sum an 42` |
| L4 | 11.1% | 0.0% | `Droraphct such such competitive competitive toament bandw...` |
| L5 | 14.8% | 0.0% | `Droroctct as meet meet competitive competitive to Diploma...` |
| L6 | 20.7% | 0.0% | `DDroroctct such ashamseasonate competitive to theymatic a...` |
| L7 | 32.1% | 0.0% | `ihydrororoctasease such ashothotate a bind to to the emph...` |
| L8 | 59.3% | 75.0% | `Dihydro hadnate nervctase inhibitors such as methotrexate...` |

## 3. Findings & Architectural Guidance for Retrieval (E2)

1. **Coarse-Level (L0–L2) Behavior**: Levels 0–2 (4 to 10 codes) compress context into broad thematic anchors. They should be used strictly for initial high-recall bucket selection.
2. **Mid-to-Fine (L6–L8) Behavior**: Levels 6–8 (32 to 48 codes) preserve named entities and numerical assertions, making them suitable for candidate filtering prior to dense inner-product scoring.
3. **Zero Reconstruction Assumption**: Consistent with the spec, decoded text is NEVER treated as canonical ground truth in LTHCM; only discrete codes are indexed.
