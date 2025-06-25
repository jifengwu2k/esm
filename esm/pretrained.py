from typing import Callable

import torch
import torch.nn as nn

from esm.models.esm3 import ESM3
from esm.models.function_decoder import FunctionTokenDecoder
from esm.models.vqvae import (
    StructureTokenDecoder,
    StructureTokenEncoder,
)
from esm.tokenization import (
    TokenizerCollection, EsmSequenceTokenizer, StructureTokenizer,
    SecondaryStructureTokenizer, SASADiscretizingTokenizer, InterProQuantizedTokenizer, ResidueAnnotationsTokenizer,
)
from esm.utils.constants import esm3 as C

ModelBuilder = Callable[[torch.device | str], nn.Module]


def ESM3_structure_encoder_v0(
        device: torch.device,
        esm3_structure_encoder_v0_pth_path: str
) -> StructureTokenEncoder:
    with torch.device(device):
        model = StructureTokenEncoder(
            d_model=1024, n_heads=1, v_heads=128, n_layers=2, d_out=128, n_codes=4096
        ).eval()
    state_dict = torch.load(
        esm3_structure_encoder_v0_pth_path,
        map_location=device,
    )
    model.load_state_dict(state_dict)
    return model


def ESM3_structure_decoder_v0(
        device: torch.device,
        esm3_structure_decoder_v0_pth_path: str
) -> StructureTokenDecoder:
    with torch.device(device):
        model = StructureTokenDecoder(d_model=1280, n_heads=20, n_layers=30).eval()
    state_dict = torch.load(
        esm3_structure_decoder_v0_pth_path,
        map_location=device,
    )
    model.load_state_dict(state_dict)
    return model


def ESM3_function_decoder_v0(
        device: torch.device,
        esm3_function_decoder_v0_pth_path: str,
        keyword_vocabulary_path: str
) -> FunctionTokenDecoder:
    with torch.device(device):
        model = FunctionTokenDecoder(
            d_model=1024,
            n_heads=8,
            n_layers=3,
            function_token_vocab_size=260,
            function_token_depth=8,
            interpro_entry_list=C.INTERPRO_ENTRY,
            keyword_vocabulary_path=keyword_vocabulary_path,
            unpack_lsh_bits=True,
            num_special_tokens=4,
            bits_per_token=8,
        ).eval()
    state_dict = torch.load(
        esm3_function_decoder_v0_pth_path,
        map_location=device,
    )
    model.load_state_dict(state_dict)
    return model


def ESM3_sm_open_v0(
        device: torch.device,
        esm3_structure_encoder_v0_pth_path: str,
        esm3_structure_decoder_v0_pth_path: str,
        esm3_function_decoder_v0_pth_path: str,
        keyword_vocabulary_path: str,
        esm3_sm_open_v1_pth_path: str
) -> ESM3:
    with torch.device(device):
        structure_encoder = ESM3_structure_encoder_v0(
            device,
            esm3_structure_encoder_v0_pth_path
        )

        structure_decoder = ESM3_structure_decoder_v0(
            device,
            esm3_structure_decoder_v0_pth_path
        )

        function_decoder = ESM3_function_decoder_v0(
            device,
            esm3_function_decoder_v0_pth_path,
            keyword_vocabulary_path
        )

        tokenizers = TokenizerCollection(
            sequence=EsmSequenceTokenizer(),
            structure=StructureTokenizer(),
            secondary_structure=SecondaryStructureTokenizer(kind="ss8"),
            sasa=SASADiscretizingTokenizer(),
            function=InterProQuantizedTokenizer(),
            residue_annotations=ResidueAnnotationsTokenizer(),
        )

        model = ESM3(
            d_model=1536,
            n_heads=24,
            v_heads=256,
            n_layers=48,
            structure_encoder=structure_encoder,
            structure_decoder=structure_decoder,
            function_decoder=function_decoder,
            tokenizers=tokenizers,
        ).eval()
    state_dict = torch.load(
        esm3_sm_open_v1_pth_path, map_location=device
    )
    model.load_state_dict(state_dict)
    return model
