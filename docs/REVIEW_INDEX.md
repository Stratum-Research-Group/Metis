# Review index

**For first-time implementation, start with [the production build prompts](BUILD_PROMPTS.md).** The review files below record the reasoning behind those prompts.

The original `.txt` prompts are paired duplicates: each `v2` file repeats its earlier brief and adds a blanket LTHCM/Dianoia submodule requirement. Treat `v2` as the later draft, but use the reviews below before implementing it. These files review the supplied documents; they do not claim that the proposed components have been built or benchmarked.

| Prompt component | New review |
| --- | --- |
| Paideia | [Training harness, labels and evaluation](paideia-review.md) |
| Phylax | [Policy and enforcement boundary](phylax-review.md) |
| Kybernetes | [Planning, contracts and outcomes](kybernetes-review.md) |
| Prosopon | [Interface, approvals and local voice](prosopon-review.md) |

The five existing components already have detailed local reviews: [Stoma](stomaimpro.md), [Dianoia](dianoiaimpro.md), [Dokime](dokimeimpro.md), [Hestia](hestiaimpro.md), and [LTHCM](LTHCMimpro.md). Their README files are alongside them. The missing piece is [the stack integration plan](stack-integration.md), which records cross-component blockers, ownership, schemas, data lifecycle, rollout order, and an end-to-end acceptance test.

**First action:** resolve the incompatible LTHCM pins, Hestia `/v1/models`, Stoma's zero-edge Dianoia path, and query timestamps. Then implement one traced, cited answer through the public interfaces before beginning Paideia training or Prosopon voice.
