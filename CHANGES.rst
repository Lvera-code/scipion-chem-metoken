=========
CHANGES
=========

0.4.1
=====
- Fixed a real bug in the GPU torch install found via an actual install
  attempt on a Colab GPU session (Tesla T4, 2026-08-21): a plain ``pip
  install torch`` resolved a build compiled against CUDA 13.0 while the
  real system ``nvcc`` was 12.8, breaking ``torch_scatter``'s
  compile-from-source step with a real ``RuntimeError`` (CUDA version
  mismatch). Fixed by detecting the real installed CUDA version and
  requesting the matching torch wheel index explicitly -- confirmed this
  produces a torch build whose own ``torch.version.cuda`` matches
  ``nvcc`` exactly. Whether ``torch_scatter`` itself then compiles
  successfully end-to-end was NOT reconfirmed after this fix (the
  validation session was lost mid-check and GPU quota prevented
  reopening one) -- flagged as the one remaining real unknown, not
  silently assumed fixed.

0.4.0
=====
- GPU support: ``USE_GPU``/``GPU_LIST`` hidden params added to
  ``ProtMeTokenCorroboration``, wired to ``CUDA_VISIBLE_DEVICES`` in
  ``runMeToken`` (the runner redirects a hardcoded ``device='cuda'`` to
  CPU only when CUDA is unavailable, no native CLI flag). Torch install +
  nvidia/triton purge are now GPU-conditional; without a GPU (this dev
  machine's case, the only branch verified here) stays exactly the
  already-verified CPU-only-wheel + purge behavior. The
  ``CUDA_VISIBLE_DEVICES`` lever itself was verified for real against
  torch on a Colab GPU session (Tesla T4): ``torch.cuda.is_available()``
  flips False/True exactly as expected; that same session confirmed
  ``nvcc`` (CUDA 12.8) is present there too, so the real caveat below does
  not bite in that specific environment. Documented caveat: a
  GPU-accelerated ``torch_scatter`` build additionally needs the CUDA
  developer toolkit (``nvcc``), not just the runtime driver -- would bite
  on a bare production host with only the runtime driver installed.

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
