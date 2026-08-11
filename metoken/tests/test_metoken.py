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

import os

from pwchem.objects import SetOfSequenceROIs
from pwchem.protocols import ProtChemPrepareReceptor, ProtDefineSeqROI
from pwem.protocols import ProtImportPdb, ProtImportSequence
from pyworkflow.tests import BaseTest, setupTestProject

from ..protocols import ProtMeTokenCorroboration

# Mismo fixture real que scipion-chem-discotope/-deepptmpred/-emngly (7c4s,
# gotcha mmCIF label_asym_id 'C' == author chain 'A').
_TEST_PDB_ID = '7c4s'
_TEST_CHAIN = 'C'


class TestMeTokenCorroboration(BaseTest):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        setupTestProject(cls)

        protImportPdb = cls.newProtocol(ProtImportPdb, inputPdbData=0, pdbId=_TEST_PDB_ID)
        cls.proj.launchProtocol(protImportPdb, wait=True)

        cls.protPrepareReceptor = cls.newProtocol(
            ProtChemPrepareReceptor, inputAtomStruct=protImportPdb.outputPdb,
            usePDBFixer=True, addRes=False, HETATM=False, rchains=True,
            chain_name='{"model": 0, "chain": "%s"}' % _TEST_CHAIN,
        )
        cls.proj.launchProtocol(cls.protPrepareReceptor, wait=True)

        protImportSeq = cls.newProtocol(
            ProtImportSequence, inputSequenceName='METOKEN_TEST_SEQ',
            inputSequenceDescription='placeholder, not used by ProtMeTokenCorroboration',
            inputRawSequence='X' * 15,
        )
        cls.proj.launchProtocol(protImportSeq, wait=True)

        inROIs = '1) Residues: {"index": "5-5", "residues": "X", "desc": "None"}'
        protDefSeqROIs = cls.newProtocol(ProtDefineSeqROI, chooseInput=0, inROIs=inROIs)
        protDefSeqROIs.inputSequence.set(protImportSeq)
        protDefSeqROIs.inputSequence.setExtended('outputSequence')
        cls.proj.launchProtocol(protDefSeqROIs, wait=True)

        sqlitePath = protDefSeqROIs.outputROIs.getFileName()
        oldItems = [roi.clone() for roi in protDefSeqROIs.outputROIs]
        os.remove(sqlitePath)
        rebuilt = SetOfSequenceROIs(filename=sqlitePath)
        for roi in oldItems:
            roi.setType('phosphorylation')
            rebuilt.append(roi)
        rebuilt.write()
        cls.protAnnotated = protDefSeqROIs

    def test(self):
        # chainId=_TEST_CHAIN ('C'), NO el default 'A': bug real de fixture
        # encontrado 2026-08-11 via 'scipion3 test' real -- ProtChemPrepareReceptor
        # filtra por el label_asym_id ('C', ver _TEST_CHAIN arriba) pero el PDB de
        # salida CONSERVA esa misma letra 'C' como su chain ID real (no la
        # renombra al author chain 'A'), confirmado inspeccionando el PDB de
        # salida real -- pedir chain_id='A' a MeToken devuelve 0 residuos.
        protMeToken = self.newProtocol(ProtMeTokenCorroboration, chainId=_TEST_CHAIN)
        protMeToken.inputROIs.set(self.protAnnotated)
        protMeToken.inputROIs.setExtended('outputROIs')
        protMeToken.inputStructure.set(self.protPrepareReceptor)
        protMeToken.inputStructure.setExtended('outputStructure')
        self.launchProtocol(protMeToken, wait=True)

        outROIs = getattr(protMeToken, 'outputROIs', None)
        self.assertIsNotNone(outROIs)
        self.assertEqual(len(outROIs), 1)

        roi = list(outROIs)[0].clone()
        # Corrida real (MeToken instalado+checkpoint descargado, verificado
        # 2026-08-11 via 'scipion3 test' real): tipo/probabilidad reales.
        self.assertIsNotNone(roi._metokenType.get())
        self.assertIsNotNone(roi._metokenProbability.get())
