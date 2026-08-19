#!/usr/bin/env python
"""Standalone runner for MeToken (informative TYPE corroboration, not a consensus engine).

A maintained, byte-for-byte vendored copy -- same policy as
``scipion-chem-deepptmpred``/``scipion-chem-emngly``: the 2 patches it
contains (Biopython three_to_one, hardcoded device='cuda') are never
rewritten from memory.

Requires torch/torch_scatter/transformers/biopython/omegaconf,
dependencies ONLY present in this plugin's dedicated conda environment.
It is invoked EXCLUSIVELY via subprocess from this plugin's protocol,
same pattern as ``scipion-chem-deepptmpred``'s runner.

## Why it exists (role in the pipeline)

``github.com/A4Bio/MeToken`` (ICLR 2025) is the most powerful structural
engine evaluated for PTM -- it consumes real backbone coordinates (N/CA/C/O)
via a 3D-kNN graph + quaternion local frames, far richer than the 4 scalars
(SASA/phi/psi/plDDT) DeepPTMPred uses -- but the PUBLISHED checkpoint is a
TYPE CLASSIFIER on ALREADY-KNOWN sites, not a site detector: confirmed in
the repo's ``model_interface.py:40`` (``valid_idx = batch['Q'] >
0 if self.hparams.with_null_ptm == 0 else ...`` -- the "Not a PTM type"
class is excluded from evaluation/training when ``with_null_ptm=0``, which
is how the published checkpoint ships). Verified with a real run against
``AF-P10636-F1-model_v4.pdb`` (Tau): at positions with NO real PTM
(prolines, glycines) it predicts types with equally high confidence -- it
CANNOT be used to decide whether a site is a PTM or not.

That is why its role here follows the same non-decisory pattern as
secretory-localization corroboration in ``scipion-chem-ptmannotation``:
purely informative TYPE corroboration on sites the consensus has ALREADY
accepted, it NEVER changes acceptance/consensus.

## Two bugs confirmed by running the repo (not assumed)

1. **``inference.py:61`` calls ``PDB.Polypeptide.three_to_one``**, removed
   from Biopython in version >=1.80 (confirmed: ``hasattr(PDB.Polypeptide,
   'three_to_one')`` -> ``False`` in Biopython 1.87) -- fails with
   ``AttributeError`` in any modern environment. Replaced here with
   ``PDB.Polypeptide.protein_letters_3to1``/``protein_letters_3to1_extended``
   (the dictionaries that DO exist in modern Biopython), monkeypatched
   onto the already-imported module -- ``inference.py`` is NOT edited
   (vendored).

2. **``src/metoken_model.py:213`` has a hardcoded ``device='cuda'``**
   (``codebook_mask = torch.ones(len(codebook), dtype=torch.int32,
   device='cuda')``, inside ``MeToken_Model.__init__``) -- makes it
   impossible to build the model on CPU. Verified: without the patch,
   ``MeToken_Model(params)`` fails with ``AssertionError: Torch not
   compiled with CUDA enabled`` on this machine (no GPU, no
   ``nvidia-smi``). It is the ONLY line in all of ``src/`` with a
   hardcoded ``device='cuda'`` (verified via grep -- every other tensor
   uses ``device=x.device``/``device=index.device``, following the input
   tensor's device). Since it sits inside an ``__init__`` (not a
   standalone, replaceable function), it is patched by wrapping
   ``torch.ones`` in a context manager active ONLY during model
   construction: if ``device='cuda'`` is requested and CUDA is not
   available, it redirects to ``'cpu'``; any other call to ``torch.ones``
   (inside or outside that block) is unaffected. ``src/metoken_model.py``
   is not edited (vendored).

Both patches verified: running WITHOUT patch 2 fails
(``AssertionError``); running WITH both patches against
``examples/Q16613.pdb`` (the repo's own example) reproduces EXACTLY the
result documented in its ``quick_inference.ipynb``
(``PTM type at the position 31 is Phosphorylation``) -- confirms the
patches do not alter the model's numerical behavior, they only make it
run in this environment.

## Detecting the "non-PTM"/"rare" class (24 real classes, not 26)

``src/constant.py::PTMtype_list`` has 26 entries: index 0 = "Not a PTM
type" (the null class, masked during training -- see above), indices
1-24 = the 24 real PTM types the model distinguishes, index 25 = "in
Rare PTM Types" (a bucket of grouped rare PTMs, not a specific
interpretable type). This runner excludes BOTH indices (0 and 25) when
looking for the highest-probability type -- "among the 24 classes", as
the task requires -- and reports the raw probability (softmax over the
full 26 classes, not renormalized) of the winning index among those 24.

## Offline (same caveat as this project's other ESM-based engines)

``AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")`` is
called TWICE in the repo (``src/metoken_model.py`` and
``src/datasets/featurizer.py``, a new tokenizer each time -- it is not
the full ESM-2 encoder, MeToken uses its own ``nn.Embedding`` trained
from scratch, ``wo_esm``, not real ESM representations despite the
attribute's name) -- downloaded from the HF Hub the first time, cacheable
locally afterward (see ``HF_HUB_OFFLINE``/``TRANSFORMERS_OFFLINE`` below).

## torch_scatter with no prebuilt wheel

``src/metoken_module.py`` imports ``torch_scatter`` (``scatter_sum``,
``scatter_softmax``, ``scatter_mean``) -- no prebuilt wheel exists for
this environment's torch/CPU/Python combination on ``data.pyg.org``
(confirmed: the wheel index only lists variants up to
``torch-2.1.0+cpu``, this machine has ``torch==2.13.0+cpu``), so
``pip install --no-build-isolation torch_scatter`` compiles from source
(C++/CPU extension, ~a few minutes on this real machine, not the ~15 min
estimated beforehand).
"""

import argparse
import contextlib
import os
import sys
from pathlib import Path

# Offline ONLY if "facebook/esm2_t33_650M_UR50D" is already in the local HF
# Hub cache -- on a machine that already downloaded it once (local dev)
# this avoids hitting the network on every run, but forcing it always and
# unconditionally (as this used to do) breaks the first run on a new
# machine with no cache (e.g. a freshly created Colab runtime):
# LocalEntryNotFoundError, with no way to ever download it. Cache layout
# verified against huggingface_hub's documentation:
# "<HF_HOME>/hub/models--<org>--<name>".
_HF_HOME = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
_ESM2_CACHE_DIR = _HF_HOME / "hub" / "models--facebook--esm2_t33_650M_UR50D"
if _ESM2_CACHE_DIR.is_dir():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import pandas as pd  # noqa: E402

OUTPUT_COLUMNS = ["position", "metoken_type", "metoken_probability"]

# PTMtype_list indices to always exclude when looking for the
# highest-probability type (see the module docstring): 0 = "Not a PTM
# type" (the null class masked during training, it never learned to fire
# reliably), 25 = "in Rare PTM Types" (a bucket of grouped rare types, not
# interpretable as a specific type).
_NULL_CLASS_INDEX = 0
_RARE_CLASS_INDEX = 25


@contextlib.contextmanager
def _force_cpu_ones():
    """Real patch 2 (see the module docstring): hardcoded
    ``torch.ones(..., device='cuda')`` in
    ``src/metoken_model.py::MeToken_Model.__init__`` (line 213).

    Active ONLY during model construction -- wraps ``torch.ones`` to
    redirect ``device='cuda'`` to ``'cpu'`` only when CUDA is not
    available, restoring the original function on exiting the block
    (never leaves the monkeypatch active longer than necessary).
    ``src/metoken_model.py`` is not edited (vendored).
    """
    import torch

    original_ones = torch.ones

    def _patched_ones(*args, **kwargs):
        if kwargs.get("device") == "cuda" and not torch.cuda.is_available():
            kwargs["device"] = "cpu"
        return original_ones(*args, **kwargs)

    torch.ones = _patched_ones
    try:
        yield
    finally:
        torch.ones = original_ones


def _patch_three_to_one() -> None:
    """Real patch 1 (see the module docstring): ``PDB.Polypeptide.three_to_one``
    removed from Biopython >=1.80, used by ``inference.py::get_seq_str``.

    Monkeypatches the attribute onto the already-imported ``Bio.PDB.Polypeptide``
    module (``inference.py`` is not edited, vendored) -- ``inference.py``
    references it as ``PDB.Polypeptide.three_to_one(...)`` inside a
    function body, resolved dynamically on each call, so the patch applies
    regardless of import order.
    """
    from Bio import PDB

    def _three_to_one(resname: str) -> str:
        from Bio.PDB.Polypeptide import protein_letters_3to1, protein_letters_3to1_extended

        if resname in protein_letters_3to1:
            return protein_letters_3to1[resname]
        if resname in protein_letters_3to1_extended:
            return protein_letters_3to1_extended[resname]
        raise KeyError(resname)

    PDB.Polypeptide.three_to_one = _three_to_one


def _load_metoken_modules(repo_dir: Path):
    """Inserts ``repo_dir`` into ``sys.path`` and imports the vendorized repo's modules.

    ``repo_dir`` is inserted at position 0 of ``sys.path`` so
    ``import src.metoken_model`` resolves to MeToken's own ``src/`` (a
    namespace package with no ``__init__.py`` at its root) -- confirmed by
    running this runner as a real script, not interactively (interactive
    mode adds the working directory to ``sys.path`` instead, which could
    produce a collision with an unrelated ``src`` package elsewhere; be
    careful when testing manually with ``python -c``).
    """
    sys.path.insert(0, str(repo_dir))
    _patch_three_to_one()

    from omegaconf import OmegaConf
    from src.constant import PTMtype_list
    from src.datasets.featurizer import featurize
    from src.metoken_model import MeToken_Model

    from inference import apply_ptm_indices, get_seq_str

    return {
        "OmegaConf": OmegaConf,
        "PTMtype_list": PTMtype_list,
        "featurize": featurize,
        "MeToken_Model": MeToken_Model,
        "apply_ptm_indices": apply_ptm_indices,
        "get_seq_str": get_seq_str,
    }


def _top_type_excluding_masked(probs_row, ptm_type_list):
    """Index/label/probability of the highest-probability type, excluding
    the null class (index 0) and the "rare" class (index 25) -- see the
    module docstring. ``probs_row`` is the 26-class softmax vector for ONE
    position.
    """
    best_idx = None
    best_prob = -1.0
    for idx, prob in enumerate(probs_row):
        if idx in (_NULL_CLASS_INDEX, _RARE_CLASS_INDEX):
            continue
        if prob > best_prob:
            best_prob = prob
            best_idx = idx
    return ptm_type_list[best_idx], float(best_prob)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Standalone MeToken runner (informative type corroboration)."
    )
    parser.add_argument("--repo-dir", required=True, help="Path to the github.com/A4Bio/MeToken clone")
    parser.add_argument("--checkpoint-path", required=True, help="Path to pretrained_model/checkpoint.ckpt")
    parser.add_argument("--pdb-path", required=True, help="Single-chain PDB (Phase 1.5)")
    parser.add_argument("--chain-id", default="A", help="Chain to read from the PDB (default 'A')")
    parser.add_argument(
        "--positions", required=True, type=int, nargs="+",
        help="1-based positions already accepted by consensus (pasa_umbral=true) to corroborate.",
    )
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args()

    repo_dir = Path(args.repo_dir)
    mods = _load_metoken_modules(repo_dir)

    protein_data = mods["get_seq_str"](args.pdb_path, chain_id=args.chain_id)
    seq_length = len(protein_data["seq"])

    # Requested positions that fall inside the sequence actually read from
    # the PDB (get_seq_str can return fewer residues than pdb_path if
    # there are gaps/non-standard residues with incomplete N/CA/C/O
    # coordinates) -- positions outside that range are skipped with a
    # warning, never fatal.
    valid_positions = [p for p in args.positions if 1 <= p <= seq_length]
    skipped = sorted(set(args.positions) - set(valid_positions))
    if skipped:
        print(
            f"[metoken_runner] {len(skipped)} position(s) out of range (the read sequence "
            f"has {seq_length} residues), skipped: {skipped}",
            file=sys.stderr,
        )

    if not valid_positions:
        pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(args.out_csv, index=False)
        return 0

    zero_based = [p - 1 for p in valid_positions]
    protein_data = mods["apply_ptm_indices"](protein_data, zero_based)
    data = mods["featurize"]([protein_data])

    import torch

    checkpoint = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
    params = mods["OmegaConf"].load(str(repo_dir / "configs" / "MeToken.yaml"))

    with _force_cpu_ones():
        model = mods["MeToken_Model"](params)
        model.load_state_dict(checkpoint)
    model.eval()

    with torch.no_grad():
        result = model(data)
    probs = result["log_probs"].softmax(dim=-1).cpu().tolist()

    rows = []
    for position in valid_positions:
        metoken_type, probability = _top_type_excluding_masked(probs[position - 1], mods["PTMtype_list"])
        rows.append({"position": position, "metoken_type": metoken_type, "metoken_probability": probability})

    pd.DataFrame(rows, columns=OUTPUT_COLUMNS).to_csv(args.out_csv, index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
