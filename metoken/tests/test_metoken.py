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

# Same real fixture as scipion-chem-discotope/-deepptmpred/-emngly (7c4s,
# mmCIF label_asym_id 'C' == author chain 'A' gotcha).
_TEST_PDB_ID = '7c4s'
_TEST_CHAIN = 'C'


class TestMeTokenCorroboration(BaseTest):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        setupTestProject(cls)

        cls.protImportPdb = cls._runImportPdb()
        cls.protPrepareReceptor = cls._runPrepareReceptorChainC(cls.protImportPdb)
        cls.protAnnotated = cls._buildSyntheticPhosphorylationRoi()
        # Run once here (real conda subprocess) -- the test_ methods below
        # only assert on its already-computed output.
        cls.protMeToken = cls._runMeTokenCorroboration(cls.protAnnotated, cls.protPrepareReceptor)

    @classmethod
    def _runImportPdb(cls):
        protImportPdb = cls.newProtocol(ProtImportPdb, inputPdbData=0, pdbId=_TEST_PDB_ID)
        cls.proj.launchProtocol(protImportPdb, wait=True)
        return protImportPdb

    @classmethod
    def _runPrepareReceptorChainC(cls, protImportPdb):
        protPrepareReceptor = cls.newProtocol(
            ProtChemPrepareReceptor, inputAtomStruct=protImportPdb.outputPdb,
            usePDBFixer=True, addRes=False, HETATM=False, rchains=True,
            chain_name='{"model": 0, "chain": "%s"}' % _TEST_CHAIN,
        )
        cls.proj.launchProtocol(protPrepareReceptor, wait=True)
        return protPrepareReceptor

    @classmethod
    def _buildSyntheticPhosphorylationRoi(cls):
        """Builds one synthetic ROI typed 'phosphorylation'.

        ``ProtDefineSeqROI`` has no way to set a PTM ``_type`` attribute on
        its own output -- a real ``SetOfSequenceROIs`` cannot be mutated in
        place once written (its rows are backed by an append-only sqlite
        file), so the only way to inject it is to materialize the item,
        delete that sqlite file, and rebuild a new set from scratch with
        the attribute added (same rebuild trick used in
        scipion-chem-emngly/scipion-chem-ptmannotation for the same
        reason).
        """
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
        return protDefSeqROIs

    @classmethod
    def _runMeTokenCorroboration(cls, protAnnotated, protPrepareReceptor):
        # chainId=_TEST_CHAIN ('C'), NOT the default 'A': ProtChemPrepareReceptor
        # filters by label_asym_id ('C', see _TEST_CHAIN above) but the
        # output PDB KEEPS that same letter 'C' as its real chain ID (it
        # does not rename it to the author chain 'A') -- requesting
        # chain_id='A' from MeToken would return 0 residues.
        protMeToken = cls.newProtocol(ProtMeTokenCorroboration, chainId=_TEST_CHAIN)
        protMeToken.inputROIs.set(protAnnotated)
        protMeToken.inputROIs.setExtended('outputROIs')
        protMeToken.inputStructure.set(protPrepareReceptor)
        protMeToken.inputStructure.setExtended('outputStructure')
        cls.proj.launchProtocol(protMeToken, wait=True)
        return protMeToken

    def _getSingleRoi(self):
        outROIs = getattr(self.protMeToken, 'outputROIs', None)
        self.assertIsNotNone(outROIs)
        self.assertEqual(len(outROIs), 1)
        return list(outROIs)[0].clone()

    def test_typeAssigned(self):
        """With MeToken installed and the checkpoint downloaded: a real
        predicted type, not None."""
        roi = self._getSingleRoi()
        self.assertIsNotNone(roi._metokenType.get())

    def test_probabilityAssigned(self):
        """With MeToken installed and the checkpoint downloaded: a real
        probability, not None."""
        roi = self._getSingleRoi()
        self.assertIsNotNone(roi._metokenProbability.get())
