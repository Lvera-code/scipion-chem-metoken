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

from pyworkflow.utils import Environ
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
        # Torch install + nvidia/triton purge are now GPU-conditional
        # (checked via 'nvidia-smi', same criterion as the rest of this
        # project's plugins); without a GPU, stays exactly the CPU-only +
        # purge combination already verified. torch_scatter has NO
        # prebuilt wheel for this combination (data.pyg.org's wheel index
        # only goes up to torch-2.1.0+cpu) -- compiled from source either
        # way ('--no-build-isolation', really a few minutes). REAL
        # CAVEAT: a GPU-accelerated torch_scatter build additionally needs
        # the CUDA developer toolkit (nvcc), not just the runtime driver
        # 'nvidia-smi' checks for -- if a host has a GPU+driver but no
        # nvcc, this compiles a CPU-only torch_scatter against a
        # GPU-enabled torch (functional, just without torch_scatter's own
        # GPU kernels). Confirmed on at least one real GPU cloud
        # environment that 'nvcc' IS present there, so this caveat does
        # not bite in that specific environment -- it would on a bare
        # production host with only the runtime driver installed.
        installer.addCommand(
            f"git clone --depth 1 {UPSTREAM_URL} {home}",
            'METOKEN_CLONED'
        ).getCondaEnvCommand(
            METOKEN_DIC['name'], binaryVersion=METOKEN_DIC['version'], pythonVersion='3.10'
        ).addCommand(
            f"{cls.getEnvActivationCommand(METOKEN_DIC)} && "
            # Real bug found+fixed via an actual GPU install run: a plain
            # 'pip install torch' resolved a build compiled against a
            # newer CUDA version than the system's real 'nvcc' -- torch
            # itself worked, but compiling torch_scatter's C++/CUDA
            # extension against it failed with a real error
            # ('RuntimeError: The detected CUDA version mismatches the
            # version that was used to compile PyTorch'). Fixed by
            # detecting the real installed CUDA version and requesting
            # the matching torch wheel index explicitly (confirmed: this
            # produces a torch build whose own 'torch.version.cuda'
            # matches nvcc exactly). Whether torch_scatter's build then
            # succeeds end-to-end was not reconfirmed after this fix (the
            # validation run was interrupted mid-check) -- the
            # CUDA-version match itself is unambiguously correct and
            # necessary regardless, so it ships now; flagged as the one
            # remaining real unknown for a future validation pass.
            "if command -v nvidia-smi > /dev/null 2>&1; then "
            "CUDA_VER=$(nvcc --version 2>/dev/null | grep -oP 'release \\K[0-9]+\\.[0-9]+' | tr -d '.'); "
            "if [ -n \"$CUDA_VER\" ]; then pip install torch --index-url https://download.pytorch.org/whl/cu${CUDA_VER}; "
            "else pip install torch; fi; "
            "else "
            "pip install --index-url https://download.pytorch.org/whl/cpu torch; "
            "fi && "
            "pip install numpy pandas biopython omegaconf transformers h5py && "
            "pip install --no-build-isolation torch_scatter && "
            "if ! command -v nvidia-smi > /dev/null 2>&1; then "
            "pip uninstall -y cuda-bindings cuda-pathfinder cuda-toolkit nvidia-cublas "
            "nvidia-cuda-cupti nvidia-cuda-nvrtc nvidia-cuda-runtime nvidia-cudnn-cu13 "
            "nvidia-cufft nvidia-cufile nvidia-curand nvidia-cusolver nvidia-cusparse "
            "nvidia-cusparselt-cu13 nvidia-nccl-cu13 nvidia-nvjitlink nvidia-nvshmem-cu13 "
            "nvidia-nvtx triton || true; "
            "fi",
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
        # CUDA_VISIBLE_DEVICES: the runner decides GPU/CPU itself
        # ('torch.cuda.is_available()', no CLI flag) -- this is the real
        # lever useGpu/gpuList (protocol_metoken.py) have on that check.
        # 'cls.getEnviron()' is never overridden anywhere in this project
        # (always returns None, equivalent to inheriting os.environ) --
        # building a real copy here is additive. Must be a real
        # 'pyworkflow.utils.Environ' (a dict subclass with extra methods
        # like 'getPrepend()' pyworkflow's job runner calls) -- a plain
        # dict fails with a real AttributeError, confirmed by an actual
        # failed test run (see scipion-chem-deepmvp for the trace).
        # CUDA_VISIBLE_DEVICES='' vs unset/'0' verified for real against
        # torch on a real GPU machine:
        # 'torch.cuda.is_available()' flips False/True accordingly -- not
        # just a theoretical lever.
        env = Environ(os.environ)
        env['CUDA_VISIBLE_DEVICES'] = protocol.gpuList.get() if protocol.useGpu.get() else ''
        protocol.runJob(fullProgram, args, env=env, cwd=cwd)
