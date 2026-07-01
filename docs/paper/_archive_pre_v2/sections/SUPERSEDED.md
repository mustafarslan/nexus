# sections/ is SUPERSEDED (2026-06-20)

`main.tex` is self-contained (it does not `\input` these files) and is the only
build target. After the 2026-06-20 canonical rewrite, the authoritative paper is
`docs/paper/main.tex`. The modular section files here are a stale mirror and may
contain pre-canonical numbers. Do not build from them. Either delete this
directory or rewrite `main.tex` to `\input` regenerated sections (Phase B).
