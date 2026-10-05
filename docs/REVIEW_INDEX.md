# Review index

**For first-time implementation, start with [the production build prompts](BUILD_PROMPTS.md).** The review files below record the reasoning behind those prompts.

The original `.txt` prompts are paired duplicates: each `v2` file repeats its earlier brief and adds a blanket LTHCM/Dianoia submodule requirement. Treat `v2` as the later draft, but use the reviews below before implementing it. These files review the supplied documents; they do not claim that the proposed components have been built or benchmarked.

| Prompt component | Monorepo Documentation |
| --- | --- |
| Paideia | [paideia/README.md](../paideia/README.md) & [REVIEW_AND_GAPS.md](../paideia/REVIEW_AND_GAPS.md) |
| Phylax | [phylax/README.md](../phylax/README.md) & [REVIEW_AND_GAPS.md](../phylax/REVIEW_AND_GAPS.md) |
| Kybernetes | [kybernetes/README.md](../kybernetes/README.md) & [REVIEW_AND_GAPS.md](../kybernetes/REVIEW_AND_GAPS.md) |
| Prosopon | [prosopon/README.md](../prosopon/README.md) & [REVIEW_AND_GAPS.md](../prosopon/REVIEW_AND_GAPS.md) |

The existing core components have detailed local documentation: [Stoma](../stoma/README.md), [Dianoia](../dianoia/README.md), [Dokime](../dokime/README.md), [Hestia](../hestia/README.md), [LTHCM](../LTHCM/README.md), [Kleroterion](../kleroterion/README.md), and [Krisis](../krisis/README.md). The overarching integration contract is [the stack integration plan](stack-integration.md).

**First action:** resolve the incompatible LTHCM pins, Hestia `/v1/models`, Stoma's zero-edge Dianoia path, and query timestamps. Then implement one traced, cited answer through the public interfaces before beginning Paideia training or Prosopon voice.
