=========
CHANGES
=========

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
