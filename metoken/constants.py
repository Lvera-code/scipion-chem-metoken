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

READ_URL = 'https://github.com/Lvera-code/scipion-chem-metoken'
UPSTREAM_URL = 'https://github.com/A4Bio/MeToken'

# Confirmado leyendo scripts/metoken_runner.py: el unico manejo de
# device/CUDA es el parche '_force_cpu_ones' (redirige a CPU si CUDA no
# esta disponible, siempre) -- sin ningun flag de CLI que exponer.
GPU_REQUIRED = False

# Licencia de MeToken (upstream): MIT, declarada en el LICENSE del repo original -- el copyright del archivo nombra a un tercero ajeno a los autores ('Corleone-Huang, 2023'), probablemente copiado de otra plantilla, pero el termino de la licencia (MIT) es permisivo de todas formas.

# A diferencia de DeepMVP/EMNGly, el checkpoint SI es automatizable: release
# real de GitHub, enlace de descarga directa (confirmado en
# PTM-Prediction/src/engines/metoken_engine.py::_validate_installation).
CHECKPOINT_ZIP_URL = 'https://github.com/A4Bio/MeToken/releases/download/1.0/pretrained_model.zip'
CHECKPOINT_FILENAME = 'checkpoint.ckpt'

NOINSTALL_WARNING = (
    "MeToken no esta instalado correctamente. Revisa que el repo se haya clonado (METOKEN_HOME) "
    'y que el checkpoint se haya descargado (automatico durante la instalacion). Corroboracion '
    'PURAMENTE INFORMATIVA (nunca decide consenso): su ausencia no bloquea el resto del pipeline.'
)
