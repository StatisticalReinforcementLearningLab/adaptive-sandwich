# Architecture Decision Records

This directory holds lightweight ADRs for decisions in this repo that are
significant enough to need a persistent record of *why*, not just *what* --
things a future contributor (including future us) shouldn't have to
re-derive from source or ticket history.

There's no tooling here, just numbered markdown files: `NNNN-short-title.md`,
numbered sequentially. Each one should have a `Status` (Proposed / Accepted /
Superseded by NNNN / Rejected), a `Context`, a `Decision`, and `Consequences`
section. Not every change needs one -- reach for this when a decision spans
multiple PRs, rejects a plausible alternative for a specific reason, or is
otherwise likely to be second-guessed later without the reasoning attached.
