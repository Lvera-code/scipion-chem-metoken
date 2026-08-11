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

_references = []  # MeToken (ICLR 2025, A4Bio/MeToken) sin bibtex propio verificado todavia -- misma politica que scipion-chem-emngly.


class Plugin(pwchemPlugin):
    """MeToken (A4Bio/MeToken, MIT) se instala clonando el repo upstream y
    construyendo un entorno conda dedicado (torch CPU-only + torch_scatter
    compilado desde fuente -- sin wheel prebuilt para esta combinacion de
    version, ver STATUS.md del proyecto hermano). El checkpoint SI se
    descarga automaticamente (release real de GitHub, enlace directo -- a
    diferencia de DeepMVP/EMNGly)."""

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

        # Clone ANTES del entorno conda (misma regla ya documentada en el
        # resto de plugins de este proyecto).
        #
        # torch CPU-only + purga nvidia/triton (mismo fix real ya aplicado
        # en scipion-chem-stackglyembed/scipion-chem-emngly). torch_scatter
        # NO tiene wheel prebuilt para esta combinacion (confirmado en
        # STATUS.md: el indice de wheels de data.pyg.org solo llega hasta
        # torch-2.1.0+cpu) -- se compila desde fuente
        # ('--no-build-isolation', real, unos pocos minutos, no ~15 como se
        # estimo antes de medirlo).
        installer.addCommand(
            f"git clone --depth 1 {UPSTREAM_URL} {home}",
            'METOKEN_CLONED'
        ).getCondaEnvCommand(
            METOKEN_DIC['name'], binaryVersion=METOKEN_DIC['version'], pythonVersion='3.10'
        ).addCommand(
            f"{cls.getEnvActivationCommand(METOKEN_DIC)} && "
            "pip install --index-url https://download.pytorch.org/whl/cpu torch && "
            "pip install numpy pandas scikit-learn biopython omegaconf transformers "
            "torch-geometric torchmetrics h5py && "
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
