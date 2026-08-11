=========
CHANGES
=========

0.2.0
=====
- Protocolo real (``ProtMeTokenCorroboration``): corrobora tipo de sitios
  PTM ya aceptados (``_metokenType``/``_metokenProbability``/
  ``_metokenTypeMatches``, mapeo tipo canonico->MeToken vendorizado desde
  ptm_annotation.py). Runner (parches Biopython three_to_one + device='cuda'
  hardcodeado) vendorizado byte-a-byte. Instalacion automatica completa
  (repo+entorno conda con torch_scatter compilado+checkpoint via release de
  GitHub) -- unico de los 3 motores estructurales de este proyecto sin
  ningun paso manual. Test real sobre 7c4s (mismo fixture que
  scipion-chem-discotope/-deepptmpred/-emngly).

0.1.0
=====
- Scaffolding inicial: estructura de plugin de Scipion generada siguiendo el
  mismo patron que los plugins de BCell-Epitope-Prediction (un plugin por
  herramienta). Sin logica de instalacion ni de protocolo todavia -- pendiente
  de la validacion end-to-end del pipeline en Colab, ver STATUS.md del
  proyecto ``PTM-Prediction``.
