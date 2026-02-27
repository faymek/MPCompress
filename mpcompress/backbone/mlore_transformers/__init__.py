"""
MLoRE Transformers Backbone for MPCompress

This module contains the MLoRE Vision Transformer implementations
migrated from RFC project for standalone operation.
"""

from .MLoRE_baseline_nocompress import MLoRE_vit_base_patch16_384 as MLoRE_vit_base_patch16_384_stage0
from .MLoRE_coding_input_featcom import MLoRE_vit_base_patch16_384 as MLoRE_vit_base_patch16_384_stage1
from .MLoRE_coding_input_featcom_mona import MLoRE_vit_base_patch16_384 as MLoRE_vit_base_patch16_384_stage2

from .heads import ConvHead, MLPHead, DEConvHead

__all__ = [
    'MLoRE_vit_base_patch16_384_stage0',
    'MLoRE_vit_base_patch16_384_stage1', 
    'MLoRE_vit_base_patch16_384_stage2',
    'ConvHead',
    'MLPHead',
    'DEConvHead',
]

