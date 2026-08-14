================================
MeToken Scipion plugin
================================

Scipion framework plugin wrapping MeToken (ICLR 2025 (OpenReview)) --
informative corroboration of PTM TYPE (PDB path) -- never decides
threshold-passing/consensus, only compares against the type already
agreed on in Phase 3.

``ProtMeTokenCorroboration`` corroborates the type of already-accepted PTM
sites with a vendorized runner, a maintained byte-for-byte copy of the
upstream inference script.

Original repo: https://github.com/A4Bio/MeToken

Citation: ICLR 2025 (OpenReview)

**MeToken license (upstream)**: MIT, declared in the original repo's
LICENSE -- the file's copyright names a third party unrelated to the
authors ('Corleone-Huang, 2023'), likely copied from another template, but
the license term (MIT) is permissive regardless.

===================
Install this plugin
===================

**Developer's version**

.. code-block::

            git clone https://github.com/Lvera-code/scipion-chem-metoken.git
            cd scipion-chem-metoken
            scipion3 installp -p . --devel
            scipion3 installb MeToken

The repo, the conda environment (CPU-only torch + ``torch_scatter``
compiled from source) and the checkpoint (a real GitHub release,
automatic download) install on their own -- unlike DeepMVP/EMNGly, there
is no manual step.

.. code-block::

            scipion3 tests metoken.tests
