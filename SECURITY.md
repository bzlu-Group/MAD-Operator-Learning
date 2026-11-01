# Checkpoint and runtime security

## Supported public environment

The supported public R4 environment uses Python 3.11 or newer and PyTorch
2.15.0 (`requirements-r4.txt`). Install it in a separate environment, not on
top of an older CUDA/torchvision/torchaudio installation. The paper's recorded
training used PyTorch 2.5.1+cu121; that historical fact remains in the protocol
and result records. It is not the recommended installation for new users.
Updating the public runtime does not retrospectively change the recorded
experiments, nor promise bitwise-identical training on a different runtime.

PyTorch 2.5.1 is affected by CVE-2025-32434 / GHSA-53q9-r3pm-6pq6. PyTorch
also reports CVE-2026-24747 / GHSA-63cw-57p8-fm3p in versions through 2.9.1.
Both concern maliciously crafted checkpoint files; an alert does not by
itself establish that a repository, machine, or experimental result was
compromised. The upstream advisories are:

- https://github.com/pytorch/pytorch/security/advisories/GHSA-53q9-r3pm-6pq6
- https://github.com/pytorch/pytorch/security/advisories/GHSA-63cw-57p8-fm3p

## Loading models and resuming training

The public R4 evaluators and plotting entry point use restricted
`weights_only=True` loading on CPU by default. A rejected restricted load
is never automatically retried with unrestricted pickle.

Full resume checkpoints contain Python/NumPy random state. Only for files
you created or whose source you independently trust, supply
`--trust-checkpoint` to a public R4 command that needs such a checkpoint.
`train_public.py --resume` requires that explicit flag. It enables pickle
loading, which can execute code even on a patched PyTorch version. Do not
use it to make an unknown downloaded file "work". Checksums detect changes
only when the expected checksum itself comes from a trusted source; they
are not a substitute for establishing trust.

The frozen `train_pinn_balanced.py`, earlier R3 programs and original
benchmark scripts preserve their historical loading behavior. Do not use
them directly with untrusted model files. Prefer the guarded public R4
entry points for the final source-term experiments. No claim is made that
every legacy workflow has been revalidated on the new runtime.

These changes address known dependency and loading risks; they do not make
arbitrary model files or executable research code safe. Keep the runtime
updated, avoid unknown checkpoints, and run reproduction in an isolated
environment without sensitive files or credentials.

## Reporting

Do not publish credentials or exploit payloads in an issue. Report an
upstream PyTorch issue using its security policy. For a repository-specific
problem, contact the authors using the contact information in the article.
