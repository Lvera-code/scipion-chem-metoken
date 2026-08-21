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
This package contains a protocol for informative PTM-type corroboration
using a local MeToken installation (structural graph classifier).
"""

import os
import subprocess

from scipion.install.funcs import InstallHelper

from pwchem import Plugin as pwchemPlugin

from .constants import CHECKPOINT_FILENAME, CHECKPOINT_ZIP_URL, METOKEN_DIC, NOINSTALL_WARNING, UPSTREAM_URL

_references = []  # MeToken (ICLR 2025, A4Bio/MeToken) has no verified BibTeX entry yet -- same policy as scipion-chem-emngly.


class Plugin(pwchemPlugin):
    """MeToken (A4Bio/MeToken, MIT) is installed by cloning the upstream
    repo and building a dedicated conda environment (CPU-only torch +
    torch_scatter compiled from source -- no prebuilt wheel exists for
    this version combination, package selection trimmed to what
    'inference.py' actually needs, not the full upstream training
    environment.yml -- see addMeTokenPackage below). The checkpoint is
    downloaded automatically (a real GitHub release, direct link)."""

    @classmethod
    def _defineVariables(cls):
        cls._defineEmVar(METOKEN_DIC['home'], cls.getEnvName(METOKEN_DIC))
        cls._defineVar(METOKEN_DIC['activation'], cls.getEnvActivationCommand(METOKEN_DIC))

    @classmethod
    def defineBinaries(cls, env):
        cls.addMeTokenPackage(env)

    @classmethod
    def addMeTokenPackage(cls, env, default=True):
        home = cls.getVar(METOKEN_DIC['home'])

        installer = InstallHelper(METOKEN_DIC['name'], packageHome=home,
                                  packageVersion=METOKEN_DIC['version'])

        # Clone BEFORE the conda environment (same rule already documented
        # across the rest of this project's plugins).
        #
        # Not installed from the upstream 'environment.yml' as-is: that
        # file is a full TRAINING environment (jupyter, wandb, tensorboard,
        # pytorch-lightning, torcheval, umap-learn, openmm, mmseqs2 --
        # none of it version-pinned at all), while this protocol only ever
        # runs 'inference.py'. Its real transitive import chain was traced
        # file-by-file (inference.py -> src.metoken_model ->
        # src.design_utils/src.metoken_module, src.datasets.featurizer,
        # src.constant) AND verified empirically via a real ablation test
        # (uninstalling each candidate package from the already-working
        # conda env, one at a time, and re-running 'scipion3 tests'):
        # scikit-learn/torch-geometric/torchmetrics are confirmed NOT
        # needed (test still passes without them -- they are only used by
        # 'model_interface.py'/'data_interface.py', training-only files
        # never imported here) and dropped; h5py MUST stay even though no
        # file on the real import path touches it directly --
        # 'src/datasets/__init__.py' pulls in 'ptm_dataset.py' (which does
        # 'import h5py') as a real transitive side effect of Python's own
        # package-import mechanism the moment
        # 'from src.datasets.featurizer import featurize' runs, confirmed
        # by a real ModuleNotFoundError traceback when it was removed.
        #
        # CPU-only torch + nvidia/triton purge (same real fix already
        # applied in scipion-chem-stackglyembed/scipion-chem-emngly).
        # torch_scatter has NO prebuilt wheel for this combination
        # (data.pyg.org's wheel index only goes up to torch-2.1.0+cpu) --
        # compiled from source ('--no-build-isolation', really a few
        # minutes, not the ~15 estimated before it was actually measured).
        installer.addCommand(
            f"git clone --depth 1 {UPSTREAM_URL} {home}",
            'METOKEN_CLONED'
        ).getCondaEnvCommand(
            METOKEN_DIC['name'], binaryVersion=METOKEN_DIC['version'], pythonVersion='3.10'
        ).addCommand(
            f"{cls.getEnvActivationCommand(METOKEN_DIC)} && "
            "pip install --index-url https://download.pytorch.org/whl/cpu torch && "
            "pip install numpy pandas biopython omegaconf transformers h5py && "
            "pip install --no-build-isolation torch_scatter && "
            "pip uninstall -y cuda-bindings cuda-pathfinder cuda-toolkit nvidia-cublas "
            "nvidia-cuda-cupti nvidia-cuda-nvrtc nvidia-cuda-runtime nvidia-cudnn-cu13 "
            "nvidia-cufft nvidia-cufile nvidia-curand nvidia-cusolver nvidia-cusparse "
            "nvidia-cusparselt-cu13 nvidia-nccl-cu13 nvidia-nvjitlink nvidia-nvshmem-cu13 "
            "nvidia-nvtx triton || true",
            'METOKEN_DEPS_INSTALLED'
        ).addCommand(
            f"cd {home} && curl -fsSL -o pretrained_model.zip {CHECKPOINT_ZIP_URL} && "
            "unzip -o pretrained_model.zip -d pretrained_model_tmp && "
            "mv pretrained_model_tmp/*/* pretrained_model/ 2>/dev/null || "
            "mv pretrained_model_tmp/* pretrained_model/ && "
            "rm -rf pretrained_model.zip pretrained_model_tmp",
            'METOKEN_CHECKPOINT_DOWNLOADED'
        ).addPackage(env, dependencies=['conda', 'git', 'curl', 'unzip'], default=default)

    @classmethod
    def validateInstallation(cls):
        """Check that this plugin's requirements are met. Returns a list of
        actionable error messages, empty if the installation is correct."""
        errors = []

        entryScript = os.path.join(cls.getMeTokenDir(), 'inference.py')
        if not os.path.isfile(entryScript):
            errors.append(f"Could not find 'inference.py' under METOKEN_HOME: '{cls.getMeTokenDir()}'.")
        elif not cls.checkCallEnv(METOKEN_DIC):
            errors.append("Activation of the MeToken conda environment failed.")

        checkpointPath = cls.getCheckpointPath()
        if not os.path.isfile(checkpointPath):
            errors.append(f"MeToken checkpoint not found: '{checkpointPath}'.")

        if errors:
            errors.append(NOINSTALL_WARNING)
        return errors

    @classmethod
    def checkCallEnv(cls, packageDic):
        actCommand = cls.getVar(packageDic['activation'])
        try:
            if 'conda' in actCommand and 'shell.bash hook' not in actCommand:
                actCommand = f'{cls.getCondaActivationCmd()}{actCommand}'
            subprocess.check_output(
                f'{actCommand} && python -c "import torch, torch_scatter, omegaconf"', shell=True
            )
            return True
        except subprocess.CalledProcessError:
            return False

    # ---------------------------------- Utils -----------------------------------

    @classmethod
    def getMeTokenDir(cls):
        return cls.getVar(METOKEN_DIC['home'])

    @classmethod
    def getCheckpointPath(cls):
        return os.path.join(cls.getMeTokenDir(), 'pretrained_model', CHECKPOINT_FILENAME)

    @classmethod
    def getRunnerScriptPath(cls):
        pluginDir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(pluginDir, 'scripts', 'metoken_runner.py')

    # ---------------------------------- Protocol functions-----------------------

    @classmethod
    def runMeToken(cls, protocol, args, cwd=None):
        activation = cls.getVar(METOKEN_DIC['activation'])
        scriptPath = cls.getRunnerScriptPath()
        fullProgram = f'MPLBACKEND=Agg {activation} && python {scriptPath}'
        protocol.runJob(fullProgram, args, env=cls.getEnviron(), cwd=cwd)
