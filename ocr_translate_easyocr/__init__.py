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

__version__ = '0.2.0'

easyocr_box_model_data = {
    'name': 'easyocr',
    'lang': [
        'af', 'ar', 'as', 'az', 'be', 'bg', 'bn', 'bs', 'ce', 'cs', 'cy', 'da', 'de', 'en', 'et', 'fr', 'ga', 'hi',
        'hr', 'hu', 'id', 'is', 'it', 'ja', 'kn', 'ko', 'ku', 'la', 'lt', 'lv', 'mi', 'mn', 'mr', 'ms', 'mt', 'ne',
        'no', 'oc', 'pi', 'pl', 'pt', 'ru', 'sk', 'sl', 'sq', 'sv', 'sw', 'ta', 'te', 'th', 'tg', 'tl', 'tr', 'ur',
        'uz', 'vi', 'zh', 'zht'
        ],
    'lang_code': 'iso1',
    'entrypoint': 'easyocr.box',
    'iso1_map': {
        'ce': 'che',
        'zh': 'ch_sim',
        'zht': 'ch_tra',
        'tg': 'tjk',
    }
}
