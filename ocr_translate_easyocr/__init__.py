###################################################################################
# ocr_translate-easyocr - a plugin for ocr_translate                              #
# Copyright (C) 2023-present Davide Grassano                                      #
#                                                                                 #
# This program is free software: you can redistribute it and/or modify            #
# it under the terms of the GNU General Public License as published by            #
# the Free Software Foundation, either version 3 of the License.                  #
#                                                                                 #
# This program is distributed in the hope that it will be useful,                 #
# but WITHOUT ANY WARRANTY; without even the implied warranty of                  #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                   #
# GNU General Public License for more details.                                    #
#                                                                                 #
# You should have received a copy of the GNU General Public License               #
# along with this program.  If not, see {http://www.gnu.org/licenses/}.           #
#                                                                                 #
# Home: https://github.com/Crivella/ocr_translate-easyocr                         #
###################################################################################
"""Plugins to enable usage of Easyocr in ocr_translate"""

__version__ = '0.1.1'

easyocr_box_model_data = {
    # Name of the model
    'name': 'easyocr',
    # List of ISO-639-1 codes supported by the model
    'lang': ['en', 'ja', 'zh', 'ko', 'vi'],
    # How the model requires the codes to be passed (one of 'iso1', 'iso2b', 'iso2t', 'iso3')
    # If the models codes only partially match or are totally different from one of the ISO standards, see iso1_map
    'lang_code': 'iso1',
    # Name of the entrypoint for the model (should match what is used in pyproject.toml)
    'entrypoint': 'easyocr.box',
    # Maps ISO-639-1 codes to the codes used by the model. Does not need to map every language, only those that are
    # different from getattr(lang: m.Language, lang_code)
    'iso1_map': {
        'ce': 'che',
        'zh': 'ch_sim',
        'zht': 'ch_tra',
        'tg': 'tjk',
    }
}
