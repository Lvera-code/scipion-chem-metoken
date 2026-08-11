# -*- coding: utf-8 -*-
# **************************************************************************
# *
# * Authors:     Enzo Sierra (enzogael57@gmail.com)
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# * This program is distributed in the hope that it will be useful,
# * but WITHOUT ANY WARRANTY; without even the implied warranty of
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# * GNU General Public License for more details.
# *
# * You should have received a copy of the GNU General Public License
# * along with this program; if not, write to the Free Software
# * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# * 02111-1307  USA
# *
# *  All comments concerning this program package may be sent to the
# *  e-mail address 'scipion@cnb.csic.es'
# *
# **************************************************************************
"""
This protocol adds a purely informative PTM-type corroboration (structural
graph classifier) to an already-accepted set of PTM consensus sites.
"""

import csv
import os

from pwchem.objects import SetOfSequenceROIs
from pwem.protocols import EMProtocol
from pyworkflow.object import Boolean, Float, String
from pyworkflow.protocol import params

from .. import Plugin as metokenPlugin

# Tipo canonico del pipeline -> etiqueta real de MeToken (vendorizado desde
# PTM-Prediction/src/engines/ptm_annotation.py::CANONICAL_TO_METOKEN_TYPE,
# verificado leyendo src/constant.py::PTMtype_list del repo real de
# MeToken). 3 tipos de este proyecto (crotonylation/glutarylation/
# citrullination) no tienen equivalente en las 24 clases reales de MeToken
# -- deliberadamente fuera de este dict.
CANONICAL_TO_METOKEN_TYPE = {
    'phosphorylation': 'Phosphorylation',
    'phosphorylation_y': 'Phosphorylation',
    'phosphorylation_st': 'Phosphorylation',
    'acetylation': 'Acetylation',
    'acetylation_k': 'Acetylation',
    'ubiquitination': 'Ubiquitination',
    'ubiquitination_k': 'Ubiquitination',
    'hydroxylation': 'Hydroxylation',
    'gamma_carboxyglutamic_acid': 'Gamma-carboxyglutamic acid',
    'lys_methylation': 'Methylation',
    'methylation_k': 'Methylation',
    'arg_methylation': 'Methylation',
    'methylation_r': 'Methylation',
    'malonylation': 'Malonylation',
    'succinylation': 'Succinylation',
    'glutathionylation': 'Glutathionylation',
    'sumoylation': 'Sumoylation',
    'sumoylation_k': 'Sumoylation',
    's_nitrosylation': 'S-nitrosylation',
    'o_linked_glycosylation': 'O-linked Glycosylation',
    'n_linked_glycosylation': 'N-linked Glycosylation',
    'glycosylation_n': 'N-linked Glycosylation',
}


class ProtMeTokenCorroboration(EMProtocol):
    """
    AI Generated:

    Adds a PURELY INFORMATIVE PTM-type corroboration to an already-accepted
    ``SetOfSequenceROIs`` (typically ``scipion-chem-ptmannotation``'s own
    output), using a local MeToken installation (a structural graph
    classifier consuming real backbone coordinates via 3D-kNN + quaternion
    local frames). NEVER a consensus engine -- the published checkpoint is
    a TYPE classifier on ALREADY-KNOWN sites, not a site detector (verified
    empirically: it predicts types with high confidence even on positions
    with no real PTM, see ``PTM-Prediction/src/engines/_metoken_runner.py``
    docstring) -- same non-decisory contract as
    ``scipion-chem-netcleave``/Kinase Library (see
    SCIPION_INTEGRATION_SPEC.md §G.1).

    Output
    ------
    outputROIs: the same ROIs as ``inputROIs``, each with ``_metokenType``
    (String, MeToken's top predicted type among its 24 real classes -- the
    null and "rare" classes are always excluded), ``_metokenProbability``
    (Float, raw softmax) and ``_metokenTypeMatches`` (Boolean, ``None`` if
    ``_type`` has no known MeToken equivalent -- e.g. crotonylation,
    glutarylation, citrullination). Does not filter.
    """

    _label = 'metoken type corroboration'

    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputROIs', params.PointerParam, pointerClass='SetOfSequenceROIs',
                       label='Accepted PTM sites: ',
                       help='Output of scipion-chem-ptmannotation\'s ProtPTMAnnotation (or any '
                            'SetOfSequenceROIs with a _type attribute per ROI).')
        form.addParam('inputStructure', params.PointerParam, pointerClass='AtomStruct',
                       label='Input structure (single chain): ',
                       help='MUST be the SAME single-chain structure the ROI positions were '
                            'derived from -- same convention as scipion-chem-deepptmpred/-emngly.')
        form.addParam('chainId', params.StringParam, default='A',
                       label='Chain ID: ',
                       help="Chain to read from the structure (MeToken's own default is 'A').")

    def _insertAllSteps(self):
        self._insertFunctionStep(self.metokenStep)
        self._insertFunctionStep(self.createOutputStep)

    # ---------------------------------- Steps -----------------------------------

    def _getRois(self):
        return [roi.clone() for roi in self.inputROIs.get()]

    def metokenStep(self):
        rois = self._getRois()
        positions = sorted({roi.getROIIdx() for roi in rois})
        if not positions:
            return

        pdbPath = os.path.abspath(self.inputStructure.get().getFileName())
        # Ruta ABSOLUTA por consistencia/defensa (mismo bug real ya
        # encontrado+corregido en scipion-chem-deepmvp/-deepptmpred/-emngly
        # -- aqui 'runMeToken' no sobreescribe cwd, asi que en principio
        # resolveria bien de todas formas, pero no vale la pena arriesgarlo).
        outCsv = os.path.abspath(self._getExtraPath('metoken_scores.csv'))
        args = (
            f'--repo-dir {metokenPlugin.getMeTokenDir()} --checkpoint-path {metokenPlugin.getCheckpointPath()} '
            f'--pdb-path {pdbPath} --chain-id {self.chainId.get()} '
            f'--positions {" ".join(str(p) for p in positions)} --out-csv {outCsv}'
        )
        try:
            metokenPlugin.runMeToken(self, args)
        except Exception as exc:  # noqa: BLE001 -- puramente informativo, nunca debe tumbar el protocolo
            self.warning(f'MeToken failed (non-fatal, degrades to no corroboration): {exc}')

    def createOutputStep(self):
        rois = self._getRois()
        if not rois:
            return

        scores = {}
        outCsv = self._getExtraPath('metoken_scores.csv')
        if os.path.isfile(outCsv):
            with open(outCsv, newline='') as fh:
                for row in csv.DictReader(fh):
                    scores[int(row['position'])] = row

        outROIs = SetOfSequenceROIs(filename=self._getPath('sequenceROIs.sqlite'))
        for roi in rois:
            row = scores.get(roi.getROIIdx())
            metokenType = row['metoken_type'] if row else None
            metokenProb = float(row['metoken_probability']) if row else None

            expected = CANONICAL_TO_METOKEN_TYPE.get(roi.getType())
            matches = None
            if expected is not None and metokenType is not None:
                matches = expected == metokenType

            roi._metokenType = String(metokenType)
            roi._metokenProbability = Float(metokenProb) if metokenProb is not None else Float(None)
            roi._metokenTypeMatches = Boolean(matches) if matches is not None else Boolean(None)
            outROIs.append(roi)

        if len(outROIs) > 0:
            self._defineOutputs(outputROIs=outROIs)
            self._defineSourceRelation(self.inputROIs, outROIs)
            self._defineSourceRelation(self.inputStructure, outROIs)

    # ---------------------------------- Validation -------------------------------

    def _validate(self):
        # MeToken es OPCIONAL (degrada, ver docstring) -- nunca bloquea el
        # lanzamiento por su ausencia.
        return []

    def _summary(self):
        summary = []
        if self.isFinished():
            outROIs = getattr(self, 'outputROIs', None)
            if outROIs is not None:
                nMismatch = sum(1 for roi in outROIs if roi._metokenTypeMatches.get() is False)
                summary.append(f'{len(outROIs)} site(s) annotated, {nMismatch} in disagreement with MeToken.')
        return summary
