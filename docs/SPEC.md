# ISOSTER Technical Specification

**This file is the specification entry point. It delegates, and the delegation
is the specification.** Agent workflow instructions name `docs/SPEC.md` as the
source-of-truth path; that contract is satisfied by the pointer table below
rather than by duplicating content here, because a second copy of the
architecture would drift from the first.

Authority, in order:

| Topic | Canonical document |
|---|---|
| Architecture, interfaces, design decisions | [`04-architecture.md`](04-architecture.md) |
| Fitting and sampling algorithm | [`03-algorithm.md`](03-algorithm.md) |
| Configuration parameters | [`02-configuration-reference.md`](02-configuration-reference.md) |
| Public API, stop codes, usage | [`01-user-guide.md`](01-user-guide.md) |
| Long-form derivations and measured performance | [`technical/`](technical/1.0-overview.md) |

Where two documents disagree, the one named above for that topic wins. Where a
number is quoted in `technical/`, the archive under
`benchmarks/draft_timings/` wins over any prose, and CI enforces it.

Keep substantive architecture and interface changes in `04-architecture.md`;
update this file only when the authority table changes.
