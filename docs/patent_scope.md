# Patent scope

This repository publishes reproducers for specific benchmark claims. The
underlying Rosenbound platform is proprietary, and this document draws the
boundary between what is reproduced here and what is not.

## Boundary statement

> *"This repository contains reproducers for specific benchmark claims. The
> underlying Rosenbound platform includes proprietary components covered by
> USPTO provisional patent AGI-PROV-01 (filed 2026-03-22, docket AGI-PROV-01,
> sole inventor Harsh Singh). The following modules are OUT OF SCOPE for this
> repository: the Differentiable Causal Inference Engine (DCIE — patent
> Claim 2, comprising Neural Feature Extractor + DCSL + IS + NCAAP), the
> Verifiable Bounded Self-Modification framework (VBSM — Claim 3), the
> Persistent Self-Improving Memory (PSIM — Claim 1), the Modular Cognitive
> Substrate (MCS — Claim 4, including IMCP and SMCE), and the Hybrid Neuro-
> Symbolic Integration mechanisms (HNSI — Claim 5, comprising EPT + BGB +
> CGETA + MCLR). Where a Rosenbound benchmark uses one of those modules in the
> closed platform, this reproducer substitutes a public equivalent (e.g.,
> AIPW / DR-ATT / IV-2SLS / Rosenbaum sensitivity for the causal Pentagon's
> non-DCIE arms; LightGBM classical for W-lane baselines; medspaCy + scispaCy
> for clinical NER). Numbers reproduced here match the corresponding claim in
> the platform WITHOUT invoking the patented mechanism."*

## Provisional patent AGI-PROV-01

- **Filed:** 2026-03-22
- **Docket:** AGI-PROV-01
- **Sole inventor:** Harsh Singh
- **Structure:** five independent claims (1-5), each a distinct inventive
  module, plus dependent claims (6-21). Claim 18 is an omnibus claim combining
  the five modules.

## Claim-by-claim boundary

Each of the five independent modules is out of scope for this repository. None
of them is implemented here in any form; where the closed platform would invoke
one, the reproducer uses a published public equivalent instead.

| Claim | Module | Status in this repository | Public equivalent used by the reproducer |
|-------|--------|---------------------------|------------------------------------------|
| 1 | **PSIM** — Persistent Self-Improving Memory | Out of scope; never implemented here | Not applicable — no self-improving memory is used |
| 2 | **DCIE** — Differentiable Causal Inference Engine | Out of scope; never implemented here | AIPW / DR-ATT / IV-2SLS / Rosenbaum sensitivity |
| 3 | **VBSM** — Verifiable Bounded Self-Modification | Out of scope; never implemented here | Not applicable — no self-modification is performed |
| 4 | **MCS** — Modular Cognitive Substrate (including IMCP and SMCE) | Out of scope; never implemented here | Not applicable — no cognitive substrate is used |
| 5 | **HNSI** — Hybrid Neuro-Symbolic Integration | Out of scope; never implemented here | medspaCy + scispaCy classical clinical NER |

Dependent claims extend their parent module and are likewise out of scope:
claims 6-8 extend PSIM, 9-11 extend DCIE, 12-14 extend VBSM, 15 extends MCS,
16/17/19 extend HNSI, 20 extends VBSM, 21 extends PSIM, and 18 is the omnibus
combination of all five.

## What this means for reproduction

The reproducers in this repository stand entirely on public methods applied to
public and credentialed-public data. A reader can confirm every headline
number here without any access to the proprietary modules above — that is the
point of the repository. The match between a reproduced number and the
platform's claim demonstrates the result is real; it does not expose how the
closed platform achieves it.
