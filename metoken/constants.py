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

DEFAULT_VERSION = '1.0'

METOKEN_DIC = {
    'name': 'MeToken',
    'version': DEFAULT_VERSION,
    'home': 'METOKEN_HOME',
    'activation': 'METOKEN_ACTIVATION_CMD',
}

UPSTREAM_URL = 'https://github.com/A4Bio/MeToken'

# Confirmed by reading scripts/metoken_runner.py: the only device/CUDA
# handling is the '_force_cpu_ones' patch (always redirects to CPU if CUDA
# is not available) -- no CLI flag to expose.
GPU_REQUIRED = False

# MeToken license (upstream): MIT, declared in the original repo's LICENSE -- the file's copyright names a third party unrelated to the authors ('Corleone-Huang, 2023'), likely copied from another template, but the license term (MIT) is permissive regardless.

# Unlike DeepMVP/EMNGly, the checkpoint IS scriptable: a real GitHub
# release, direct download link.
CHECKPOINT_ZIP_URL = 'https://github.com/A4Bio/MeToken/releases/download/1.0/pretrained_model.zip'
CHECKPOINT_FILENAME = 'checkpoint.ckpt'

NOINSTALL_WARNING = (
    "MeToken is not installed correctly. Check that the repo has been cloned (METOKEN_HOME) "
    'and that the checkpoint has been downloaded (automatic during installation). PURELY '
    'INFORMATIVE corroboration (never decides consensus): its absence does not block the rest '
    'of the pipeline.'
)
