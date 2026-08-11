================================
MeToken Scipion plugin
================================

Scipion framework plugin wrapping MeToken (ICLR 2025 (OpenReview)) --
corroboracion informativa del TIPO de PTM (Camino PDB) -- nunca decide pasa_umbral/consenso, solo compara contra el tipo ya consensuado en Fase 3.

**Estado: protocolo real implementado, pendiente de instalacion+test real**
(ver ``PTM-Prediction/STATUS.md``, entrada 2026-08-11). ``ProtMeTokenCorroboration``
corrobora el tipo de sitios PTM ya aceptados con un runner vendorizado
(identico al ya validado end-to-end en el pipeline standalone).

Repo original: https://github.com/A4Bio/MeToken

Cita: ICLR 2025 (OpenReview)

**Licencia de MeToken (upstream)**: MIT, declarada en el LICENSE del repo original -- el copyright del archivo nombra a un tercero ajeno a los autores ('Corleone-Huang, 2023'), probablemente copiado de otra plantilla, pero el termino de la licencia (MIT) es permisivo de todas formas.

===================
Install this plugin
===================

**Developer's version**

.. code-block::

            git clone https://github.com/Lvera-code/scipion-chem-metoken.git
            cd scipion-chem-metoken
            scipion3 installp -p . --devel
            scipion3 installb MeToken

El repo, el entorno conda (torch CPU-only + ``torch_scatter`` compilado
desde fuente) y el checkpoint (release real de GitHub, descarga automatica)
se instalan solos -- a diferencia de DeepMVP/EMNGly, no hay ningun paso
manual.

.. code-block::

            scipion3 tests metoken.tests
