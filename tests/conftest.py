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
"""Fixtures for tests."""

import numpy as np
import pytest
from PIL import Image

from ocr_translate_easyocr import plugin as easyocr

strings = [
    'This is a test string.',
    'This is a test string.\nWith a newline.',
    'This is a test string.\nWith a newline.\nAnd another.',
    'This is a test string.? With a special break character.',
    'This is a test string.? With a special break character.\nAnd a newline.',
    'String with a dash-newline brok-\nen word.'
]
ids = [
    'simple',
    'newline',
    'newlines',
    'breakchar',
    'breakchar_newline',
    'dash_newline'
]

@pytest.fixture(params=strings, ids=ids)
def string(request):
    """String to perform TSL on."""
    return request.param

@pytest.fixture()
def batch_string(string):
    """Batched string to perform TSL on."""
    return [string, string, string]

@pytest.fixture(scope='session')
def image_pillow():
    """Random Pillow image."""
    npimg = np.random.randint(0,255,(25,25,3), dtype=np.uint8)
    return Image.fromarray(npimg)

@pytest.fixture()
def easyocr_model() -> easyocr.EasyOCRBoxModel:
    """OCRBoxModel database object."""
    easyocr_model_dict = {
        'name': 'easyocr',
        'language_format': 'iso1',
        'entrypoint': 'easyocr.box'
    }

    return easyocr.EasyOCRBoxModel(**easyocr_model_dict)

@pytest.fixture(scope='function')
def mock_called(request):
    """Generic mock function to check if it was called."""
    def mock_call(*args, **kwargs): # pylint: disable=inconsistent-return-statements
        mock_call.called = True
        mock_call.args = args
        mock_call.kwargs = kwargs

        if hasattr(request, 'param'):
            return request.param

    if hasattr(request, 'param'):
        mock_call.expected = request.param

    return mock_call
