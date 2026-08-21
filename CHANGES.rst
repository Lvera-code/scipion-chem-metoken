=========
CHANGES
=========

0.4.0
=====
- GPU support: ``USE_GPU``/``GPU_LIST`` hidden params added to
  ``ProtMeTokenCorroboration``, wired to ``CUDA_VISIBLE_DEVICES`` in
  ``runMeToken`` (the runner redirects a hardcoded ``device='cuda'`` to
  CPU only when CUDA is unavailable, no native CLI flag). Torch install +
  nvidia/triton purge are now GPU-conditional; without a GPU (this dev
  machine's case, the only branch verified here) stays exactly the
  already-verified CPU-only-wheel + purge behavior. Documented a real
  caveat: a GPU-accelerated ``torch_scatter`` build additionally needs the
  CUDA developer toolkit (``nvcc``), not just the runtime driver -- not
  verified on this GPU-less machine.

0.3.0
=====
- Install list trimmed to what ``inference.py``'s real transitive import
  chain needs (verified via a real ablation test): dropped
  scikit-learn/torch-geometric/torchmetrics (training-only, confirmed
  unused), kept h5py (transitively required via
  ``src/datasets/__init__.py``, missed by a naive per-file import trace).
  Removed unused ``READ_URL`` constant. Test file split into per-behavior
  methods instead of one ``setUpClass`` blob.

0.2.0
=====
- Real protocol (``ProtMeTokenCorroboration``): corroborates the type of
  already-accepted PTM sites (``_metokenType``/``_metokenProbability``/
  ``_metokenTypeMatches``, canonical-type->MeToken mapping). Runner
  (Biopython three_to_one patches + hardcoded device='cuda') vendorized
  byte-for-byte. Fully automatic installation (repo + conda environment
  with torch_scatter compiled from source + checkpoint via a GitHub
  release) -- the only one of this project's 3 structural engines with no
  manual step. Real test on 7c4s (same fixture as
  scipion-chem-discotope/-deepptmpred/-emngly).

0.1.0
=====
- Initial scaffolding: Scipion plugin structure generated following the
  same one-plugin-per-tool pattern used across this project's other
  plugins. No installation or protocol logic yet.
