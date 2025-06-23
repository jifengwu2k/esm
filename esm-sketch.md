`/esm/esm/layers/geom_attention.py`:

```python
from math import sqrt
import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F

class GeometricReasoningOriginalImpl(nn.Module):

    def __init__(self, c_s: int, v_heads: int, num_vector_messages: int=1, mask_and_zero_frameless: bool=True, divide_residual_by_depth: bool=False, bias: bool=False):...
        super().__init__()
        self.c_s = c_s
        self.v_heads = v_heads
        self.num_vector_messages = num_vector_messages
        self.mask_and_zero_frameless = mask_and_zero_frameless
        self.s_norm = nn.LayerNorm(c_s, bias=bias)
        dim_proj = 4 * self.v_heads * 3 + self.v_heads * 3 * self.num_vector_messages
        self.proj = nn.Linear(c_s, dim_proj, bias=bias)
        channels_out = self.v_heads * 3 * self.num_vector_messages
        self.out_proj = nn.Linear(channels_out, c_s, bias=bias)
        self.distance_scale_per_head = nn.Parameter(torch.zeros(self.v_heads))
        self.rotation_scale_per_head = nn.Parameter(torch.zeros(self.v_heads))

    def forward(self, s, affine, affine_mask, sequence_id, chain_id):
        if sequence_id is None:
            sequence_id = torch.zeros_like(s[..., 0], dtype=torch.int64)
        attn_bias = sequence_id.unsqueeze(-1) == sequence_id.unsqueeze(-2)
        attn_bias = attn_bias.unsqueeze(1).float()
        attn_bias = attn_bias.masked_fill(~affine_mask[:, None, None, :], torch.finfo(attn_bias.dtype).min)
        chain_id_mask = chain_id.unsqueeze(1) != chain_id.unsqueeze(2)
        attn_bias = attn_bias.masked_fill(chain_id_mask.unsqueeze(1), torch.finfo(s.dtype).min)
        ns = self.s_norm(s)
        (vec_rot, vec_dist) = self.proj(ns).split([self.v_heads * 2 * 3 + self.v_heads * 3 * self.num_vector_messages, self.v_heads * 2 * 3], dim=-1)
        (query_rot, key_rot, value) = affine.rot[..., None].apply(rearrange(vec_rot, '... (h c) -> ... h c', c=3)).split([self.v_heads, self.v_heads, self.v_heads * self.num_vector_messages], dim=-2)
        (query_dist, key_dist) = affine[..., None].apply(rearrange(vec_dist, '... (h c) -> ... h c', c=3)).chunk(2, dim=-2)
        query_dist = rearrange(query_dist, 'b s h d -> b h s 1 d')
        key_dist = rearrange(key_dist, 'b s h d -> b h 1 s d')
        query_rot = rearrange(query_rot, 'b s h d -> b h s d')
        key_rot = rearrange(key_rot, 'b s h d -> b h d s')
        value = rearrange(value, 'b s (h m) d -> b h s (m d)', m=self.num_vector_messages)
        distance_term = (query_dist - key_dist).norm(dim=-1) / sqrt(3)
        rotation_term = query_rot.matmul(key_rot) / sqrt(3)
        distance_term_weight = rearrange(F.softplus(self.distance_scale_per_head), 'h -> h 1 1')
        rotation_term_weight = rearrange(F.softplus(self.rotation_scale_per_head), 'h -> h 1 1')
        attn_weight = rotation_term * rotation_term_weight - distance_term * distance_term_weight
        if attn_bias is not None:
            s_q = attn_weight.size(2)
            s_k = attn_weight.size(3)
            _s_q = max(0, attn_bias.size(2) - s_q)
            _s_k = max(0, attn_bias.size(3) - s_k)
            attn_bias = attn_bias[:, :, _s_q:, _s_k:]
            attn_weight = attn_weight + attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_out = attn_weight.matmul(value)
        attn_out = affine.rot[..., None].invert().apply(rearrange(attn_out, 'b h s (m d) -> b s (h m) d', m=self.num_vector_messages))
        attn_out = rearrange(attn_out, 'b s (h m) d -> b s (h m d)', m=self.num_vector_messages)
        if self.mask_and_zero_frameless:
            attn_out = attn_out.masked_fill(~affine_mask[..., None], 0.0)
        s = self.out_proj(attn_out)
        return s
```

`/esm/esm/tokenization/tokenizer_base.py`:

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class EsmTokenizerBase(Protocol):
    mask_token: str
    mask_token_id: int
    bos_token: str
    bos_token_id: int
    eos_token: str
    eos_token_id: int
    pad_token: str
    pad_token_id: int
    chain_break_token: str
    chain_break_token_id: int

    def encode(self, *args, **kwargs):
        ...

    def decode(self, *args, **kwargs):
        ...

    @property
    def all_token_ids(self):
        ...

    @property
    def special_token_ids(self):
        ...
```

`/esm/esm/tokenization/sequence_tokenizer.py`:

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast
from esm.tokenization.tokenizer_base import EsmTokenizerBase
from esm.utils.constants import esm3 as C

class EsmSequenceTokenizer(PreTrainedTokenizerFast, EsmTokenizerBase):
    model_input_names = ['sequence_tokens', 'attention_mask']

    def __init__(self, unk_token='<unk>', cls_token='<cls>', pad_token='<pad>', mask_token='<mask>', eos_token='<eos>', chain_break_token='|', **kwargs):
        all_tokens = C.SEQUENCE_VOCAB
        token_to_id = {tok: ind for (ind, tok) in enumerate(all_tokens)}
        bpe = BPE(token_to_id, merges=[], unk_token=unk_token)
        tokenizer = Tokenizer(bpe)
        special_tokens = [cls_token, pad_token, mask_token, eos_token, chain_break_token]
        self.cb_token = chain_break_token
        additional_special_tokens = [chain_break_token]
        tokenizer.add_special_tokens(special_tokens)
        tokenizer.post_processor = TemplateProcessing(single='<cls> $A <eos>', special_tokens=[('<cls>', tokenizer.token_to_id('<cls>')), ('<eos>', tokenizer.token_to_id('<eos>'))])
        super().__init__(tokenizer_object=tokenizer, unk_token=unk_token, cls_token=cls_token, pad_token=pad_token, mask_token=mask_token, eos_token=eos_token, additional_special_tokens=additional_special_tokens, **kwargs)

    @property
    def bos_token(self):...

    @property
    def bos_token_id(self):
        return self.cls_token_id

    @property
    def cls_token(self):
        return self._get_token('cls_token')

    @property
    def cls_token_id(self):
        return self._get_token_id(self.cls_token)

    @property
    def eos_token(self):
        return self._get_token('eos_token')

    @property
    def eos_token_id(self):
        return self._get_token_id(self.eos_token)

    @property
    def mask_token(self):
        return self._get_token('mask_token')

    @property
    def mask_token_id(self):
        return self._get_token_id(self.mask_token)

    @property
    def pad_token(self):
        return self._get_token('pad_token')

    @property
    def pad_token_id(self):
        return self._get_token_id(self.pad_token)

    @property
    def chain_break_token(self):...

    @property
    def chain_break_token_id(self):...

    @property
    def all_token_ids(self):
        return list(range(self.vocab_size))

    @property
    def special_token_ids(self):
        return self.all_special_ids

    def _get_token_id(self, token) -> int:
        token_id = self.convert_tokens_to_ids(token)
        assert isinstance(token_id, int)
        return token_id

    def _get_token(self, token_name: str) -> str:
        token_str = self.__getattr__(token_name)
        assert isinstance(token_str, str)
        return token_str
```

`/esm/esm/__init__.py`:

```python
__version__ = '3.2.0'
```

`/esm/esm/layers/regression_head.py`:

```python
import torch.nn as nn

def RegressionHead(d_model: int, output_dim: int, hidden_dim: int | None=None) -> nn.Module:...
    hidden_dim = hidden_dim if hidden_dim is not None else d_model
    return nn.Sequential(nn.Linear(d_model, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, output_dim))
```

`/esm/esm/tokenization/structure_tokenizer.py`:

```python
from esm.tokenization.tokenizer_base import EsmTokenizerBase
from esm.utils.constants import esm3 as C

class StructureTokenizer(EsmTokenizerBase):

    def __init__(self, codebook_size: int=C.VQVAE_CODEBOOK_SIZE):
        self.vq_vae_special_tokens = {'MASK': codebook_size, 'EOS': codebook_size + 1, 'BOS': codebook_size + 2, 'PAD': codebook_size + 3, 'CHAINBREAK': codebook_size + 4}

    def mask_token(self) -> str:...

    @property
    def mask_token_id(self) -> int:
        return self.vq_vae_special_tokens['MASK']

    def bos_token(self) -> str:...

    @property
    def bos_token_id(self) -> int:
        return self.vq_vae_special_tokens['BOS']

    def eos_token(self) -> str:...

    @property
    def eos_token_id(self) -> int:
        return self.vq_vae_special_tokens['EOS']

    def pad_token(self) -> str:...

    @property
    def pad_token_id(self) -> int:
        return self.vq_vae_special_tokens['PAD']

    def chain_break_token(self) -> str:...

    @property
    def chain_break_token_id(self) -> int:...

    @property
    def all_token_ids(self):
        return list(range(C.VQVAE_CODEBOOK_SIZE + len(self.vq_vae_special_tokens)))

    @property
    def special_token_ids(self):
        return self.vq_vae_special_tokens.values()

    def encode(self, *args, **kwargs):...

    def decode(self, *args, **kwargs):...
```

`/esm/esm/utils/structure/affine3d.py`:

```python
from __future__ import annotations
import typing as T
from dataclasses import dataclass
import torch
from typing_extensions import Self
from esm.utils.misc import fp32_autocast_context

@T.runtime_checkable
class Rotation(T.Protocol):

    @classmethod
    def identity(cls, shape: tuple[int, ...], **tensor_kwargs) -> Self:
        ...

    @classmethod
    def random(cls, shape: tuple[int, ...], **tensor_kwargs) -> Self:
        ...

    def __getitem__(self, idx: T.Any) -> Self:
        ...

    @property
    def tensor(self) -> torch.Tensor:...

    @property
    def shape(self) -> torch.Size:...

    def as_matrix(self) -> RotationMatrix:
        ...

    def compose(self, other: Self) -> Self:...

    def convert_compose(self, other: Self) -> Self:...

    def apply(self, p: torch.Tensor) -> torch.Tensor:...

    def invert(self) -> Self:
        ...

    @property
    def dtype(self) -> torch.dtype:...

    @property
    def device(self) -> torch.device:
        return self.tensor.device

    @property
    def requires_grad(self) -> bool:...

    @classmethod
    def _from_tensor(cls, t: torch.Tensor) -> Self:...

    def to(self, **kwargs) -> Self:...

    def detach(self, *args, **kwargs) -> Self:...

    def tensor_apply(self, func) -> Self:...

class RotationMatrix(Rotation):

    def __init__(self, rots: torch.Tensor):
        if rots.shape[-1] == 9:
            rots = rots.unflatten(-1, (3, 3))
        assert rots.shape[-1] == 3
        assert rots.shape[-2] == 3
        self._rots = rots.to(torch.float32)

    @classmethod
    def identity(cls, shape, **tensor_kwargs):
        rots = torch.eye(3, **tensor_kwargs)
        rots = rots.view(*[1 for _ in range(len(shape))], 3, 3)
        rots = rots.expand(*shape, -1, -1)
        return cls(rots)

    @classmethod
    def random(cls, shape, **tensor_kwargs):...

    def __getitem__(self, idx: T.Any) -> RotationMatrix:
        indices = (idx,) if isinstance(idx, int) or idx is None else tuple(idx)
        return RotationMatrix(self._rots[indices + (slice(None), slice(None))])

    @property
    def shape(self) -> torch.Size:
        return self._rots.shape[:-2]

    def as_matrix(self) -> RotationMatrix:
        return self

    def compose(self, other: RotationMatrix) -> RotationMatrix:
        with fp32_autocast_context(self._rots.device.type):
            return RotationMatrix(self._rots @ other._rots)

    def convert_compose(self, other: Rotation):...

    def apply(self, p: torch.Tensor) -> torch.Tensor:
        with fp32_autocast_context(self.device.type):
            if self._rots.shape[-3] == 1:
                return p @ self._rots.transpose(-1, -2).squeeze(-3)
            else:
                return torch.einsum('...ij,...j', self._rots, p)

    def invert(self) -> RotationMatrix:
        return RotationMatrix(self._rots.transpose(-1, -2))

    @property
    def tensor(self) -> torch.Tensor:
        return self._rots.flatten(-2)

    def to_3x3(self) -> torch.Tensor:...

    @staticmethod
    def from_graham_schmidt(x_axis: torch.Tensor, xy_plane: torch.Tensor, eps: float=1e-12) -> ...:
        return RotationMatrix(_graham_schmidt(x_axis, xy_plane, eps))

@dataclass(frozen=True)
class Affine3D:
    trans: torch.Tensor
    rot: Rotation

    def __post_init__(self):
        assert self.trans.shape[:-1] == self.rot.shape

    @staticmethod
    def identity(shape_or_affine: T.Union[tuple[int, ...], 'Affine3D'], rotation_type: T.Type[Rotation]=RotationMatrix, **tensor_kwargs):
        if isinstance(shape_or_affine, Affine3D):...
        else:
            kwargs = tensor_kwargs
            shape = shape_or_affine
        return Affine3D(torch.zeros((*shape, 3), **kwargs), rotation_type.identity(shape, **kwargs))

    @staticmethod
    def random(shape: tuple[int, ...], std: float=1, rotation_type: T.Type[Rotation]=RotationMatrix, **tensor_kwargs) -> ...:...

    def __getitem__(self, idx: T.Any) -> 'Affine3D':
        indices = (idx,) if isinstance(idx, int) or idx is None else tuple(idx)
        return Affine3D(trans=self.trans[indices + (slice(None),)], rot=self.rot[idx])

    @property
    def shape(self) -> torch.Size:
        return self.trans.shape[:-1]

    @property
    def dtype(self) -> torch.dtype:
        return self.trans.dtype

    @property
    def device(self) -> torch.device:
        return self.trans.device

    @property
    def requires_grad(self) -> bool:...

    def to(self, **kwargs) -> 'Affine3D':...

    def detach(self, *args, **kwargs) -> 'Affine3D':...

    def tensor_apply(self, func) -> 'Affine3D':...

    def as_matrix(self):
        return Affine3D(trans=self.trans, rot=self.rot.as_matrix())

    def compose(self, other: 'Affine3D', autoconvert: bool=False):
        rot = self.rot
        new_rot = (rot.convert_compose if autoconvert else rot.compose)(other.rot)
        new_trans = rot.apply(other.trans) + self.trans
        return Affine3D(trans=new_trans, rot=new_rot)

    def compose_rotation(self, other: Rotation, autoconvert: bool=False):...

    def scale(self, v: torch.Tensor | float):...

    def mask(self, mask: torch.Tensor, with_zero=False):
        if with_zero:...
        else:
            identity = self.identity(self.shape, rotation_type=type(self.rot), device=self.device, dtype=self.dtype).tensor
            return Affine3D.from_tensor(identity.where(mask[..., None], self.tensor))

    def apply(self, p: torch.Tensor) -> torch.Tensor:
        return self.rot.apply(p) + self.trans

    def invert(self):
        inv_rot = self.rot.invert()
        return Affine3D(trans=-inv_rot.apply(self.trans), rot=inv_rot)

    @property
    def tensor(self) -> torch.Tensor:
        return torch.cat([self.rot.tensor, self.trans], dim=-1)

    @staticmethod
    def from_tensor(t: torch.Tensor) -> 'Affine3D':
        match t.shape[-1]:
            case 4:...
            case 12:
                trans = t[..., -3:]
                rot = RotationMatrix(t[..., :-3].unflatten(-1, (3, 3)))
            case _:...
        return Affine3D(trans, rot)

    @staticmethod
    def from_tensor_pair(t: torch.Tensor, r: torch.Tensor) -> 'Affine3D':...

    @staticmethod
    def from_graham_schmidt(neg_x_axis: torch.Tensor, origin: torch.Tensor, xy_plane: torch.Tensor, eps: float=1e-10):
        x_axis = origin - neg_x_axis
        xy_plane = xy_plane - origin
        return Affine3D(trans=origin, rot=RotationMatrix.from_graham_schmidt(x_axis, xy_plane, eps))

    @staticmethod
    def cat(affines: list['Affine3D'], dim: int=0):...

def _graham_schmidt(x_axis: torch.Tensor, xy_plane: torch.Tensor, eps: float=1e-12):
    with fp32_autocast_context(x_axis.device.type):
        e1 = xy_plane
        denom = torch.sqrt((x_axis ** 2).sum(dim=-1, keepdim=True) + eps)
        x_axis = x_axis / denom
        dot = (x_axis * e1).sum(dim=-1, keepdim=True)
        e1 = e1 - x_axis * dot
        denom = torch.sqrt((e1 ** 2).sum(dim=-1, keepdim=True) + eps)
        e1 = e1 / denom
        e2 = torch.cross(x_axis, e1, dim=-1)
        rots = torch.stack([x_axis, e1, e2], dim=-1)
        return rots

def build_affine3d_from_coordinates(coords: torch.Tensor) -> ...:
    _MAX_SUPPORTED_DISTANCE = 1000000.0
    coord_mask = torch.all(torch.all(torch.isfinite(coords) & (coords < _MAX_SUPPORTED_DISTANCE), dim=-1), dim=-1)

    def atom3_to_backbone_affine(bb_positions: torch.Tensor) -> Affine3D:
        (N, CA, C) = bb_positions.unbind(dim=-2)
        return Affine3D.from_graham_schmidt(C, CA, N)
    coords = coords.clone().float()
    coords[~coord_mask] = 0
    average_per_n_ca_c = coords.masked_fill(~coord_mask[..., None, None], 0).sum(1) / (coord_mask.sum(-1)[..., None, None] + 1e-08)
    affine_from_average = atom3_to_backbone_affine(average_per_n_ca_c.float()).as_matrix()
    (B, S, _, _) = coords.shape
    assert isinstance(B, int)
    assert isinstance(S, int)
    affine_rot_mats = affine_from_average.rot.tensor[..., None, :].expand(B, S, 9)
    affine_trans = affine_from_average.trans[..., None, :].expand(B, S, 3)
    identity_rot = RotationMatrix.identity((B, S), dtype=torch.float32, device=coords.device, requires_grad=False)
    affine_rot_mats = affine_rot_mats.where(coord_mask.any(-1)[..., None, None], identity_rot.tensor)
    black_hole_affine = Affine3D(affine_trans, RotationMatrix(affine_rot_mats))
    affine = atom3_to_backbone_affine(coords.float())
    affine = Affine3D.from_tensor(affine.tensor.where(coord_mask[..., None], black_hole_affine.tensor))
    return (affine, coord_mask)
```

`/esm/esm/utils/function/encode_decode.py`:

```python
import re
from typing import Sequence
import torch
from esm.models.function_decoder import FunctionTokenDecoder, merge_annotations
from esm.tokenization.function_tokenizer import InterProQuantizedTokenizer
from esm.tokenization.residue_tokenizer import ResidueAnnotationsTokenizer
from esm.utils.constants import esm3 as C
from esm.utils.types import FunctionAnnotation

def encode_function_annotations(sequence: str, function_annotations: Sequence[FunctionAnnotation], function_tokens_tokenizer: InterProQuantizedTokenizer, residue_annotations_tokenizer: ResidueAnnotationsTokenizer, add_special_tokens: bool=True) -> tuple[torch.Tensor, torch.Tensor]:...

def decode_function_tokens(function_token_ids: torch.Tensor, function_token_decoder: FunctionTokenDecoder, function_tokens_tokenizer: InterProQuantizedTokenizer, decoder_annotation_threshold: float=0.1, annotation_min_length: int | None=5, annotation_gap_merge_max: int | None=3) -> list[FunctionAnnotation]:...

def decode_residue_annotation_tokens(residue_annotations_token_ids: torch.Tensor, residue_annotations_tokenizer: ResidueAnnotationsTokenizer, annotation_min_length: int | None=5, annotation_gap_merge_max: int | None=3) -> list[FunctionAnnotation]:...
```

`/esm/esm/sdk/api.py`:

```python
from __future__ import annotations
from abc import ABC
from copy import deepcopy
from typing import List, Sequence
import attr
import torch
from attr import asdict, define
import esm.utils.constants.api as C
from esm.tokenization import TokenizerCollectionProtocol, get_esm3_model_tokenizers
from esm.utils import encoding
from esm.utils.constants.models import ESM3_OPEN_SMALL
from esm.utils.misc import get_chainbreak_boundaries_from_sequence
from esm.utils.structure.protein_chain import ProteinChain
from esm.utils.structure.protein_complex import ProteinComplex
from esm.utils.types import FunctionAnnotation, PathOrBuffer

class ProteinType(ABC):
    ...

@define
class ESMProtein(ProteinType):
    sequence: str | None = None
    secondary_structure: str | None = None
    sasa: list[float | None] | None = None
    function_annotations: list[FunctionAnnotation] | None = None
    coordinates: torch.Tensor | None = None
    plddt: torch.Tensor | None = None
    ptm: torch.Tensor | None = None
    potential_sequence_of_concern: bool = False

    def __len__(self):...

    @classmethod
    def from_pdb(cls, path: PathOrBuffer, chain_id: str='detect', id: str | None=None, is_predicted: bool=False) -> ...:...

    @classmethod
    def from_protein_chain(cls, protein_chain: ProteinChain, with_annotations: bool=False) -> ...:...

    @classmethod
    def from_protein_complex(cls, protein_complex: ProteinComplex, with_annotations: bool=False) -> ...:...

    def to_pdb(self, pdb_path: PathOrBuffer) -> None:
        protein_complex = self.to_protein_complex().infer_oxygen()
        protein_complex.to_pdb(pdb_path)

    def to_pdb_string(self) -> str:...

    def to_protein_chain(self) -> ProteinChain:...

    def to_protein_complex(self, copy_annotations_from_ground_truth: ProteinComplex | None=None) -> ...:
        assert self.sequence is not None, ...
        assert self.coordinates is not None, ...
        coords = self.coordinates.to('cpu').numpy()
        chain_boundaries = get_chainbreak_boundaries_from_sequence(self.sequence)
        if copy_annotations_from_ground_truth is not None:...
        else:
            gt_chains = None
        pred_chains = []
        for (i, (start, end)) in enumerate(chain_boundaries):
            pred_chain = ProteinChain.from_atom37(atom37_positions=coords[start:end], sequence=self.sequence[start:end], chain_id=gt_chains[i].chain_id if gt_chains is not None else None, entity_id=gt_chains[i].entity_id if gt_chains is not None else None)
            pred_chains.append(pred_chain)
        return ProteinComplex.from_chains(pred_chains)

    def copy(self) -> 'ESMProtein':...

@define
class ESMProteinTensor(ProteinType):
    sequence: torch.Tensor | None = None
    structure: torch.Tensor | None = None
    secondary_structure: torch.Tensor | None = None
    sasa: torch.Tensor | None = None
    function: torch.Tensor | None = None
    residue_annotations: torch.Tensor | None = None
    coordinates: torch.Tensor | None = None
    potential_sequence_of_concern: bool = False

    def _detect_attribute(self, func, msg):
        mapped = {k: func(k, v) for (k, v) in asdict(self).items() if isinstance(v, torch.Tensor)}
        s = set(mapped.values())
        if len(s) <= 0:...
        if len(s) != 1:...
        return next(iter(s))

    def __len__(self) -> int:
        l = self._detect_attribute(lambda _, x: x.size(0), 'length')
        return l if l is not None else 0

    @property
    def device(self) -> str | torch.device:
        d = self._detect_attribute(lambda _, x: x.device, 'device')
        assert d is not None
        return d

    def to(self, device_or_dtype: str | torch.device | torch.dtype) -> ESMProteinTensor:

        def _to(name):
            v = getattr(self, name)
            if v is not None and isinstance(v, torch.Tensor):
                setattr(self, name, v.to(device_or_dtype))
        for n in attr.fields(ESMProteinTensor):
            _to(n.name)
        return self

    @classmethod
    def empty(cls, length: int, tokenizers: TokenizerCollectionProtocol | None=None, device: torch.device | str='cpu') -> ...:...

    def copy(self) -> ESMProteinTensor:...

@define
class ESMProteinError(Exception, ProteinType):
    error_code: int
    error_msg: str

@define
class GenerationConfig:
    track: str = ''
    invalid_ids: Sequence[int] = []
    schedule: str = attr.field(validator=attr.validators.in_(['cosine', 'linear']), default='cosine')
    strategy: str = attr.field(validator=attr.validators.in_(['random', 'entropy']), default='random')
    num_steps: int = 20
    temperature: float = 1.0
    temperature_annealing: bool = True
    top_p: float = 1.0
    condition_on_coordinates_only: bool = True
    only_compute_backbone_rmsd: bool = False

    def use_entropy_based_unmasking_strategy(self):...

    def use_generative_unmasking_strategy(self):...

@define
class MSA:
    sequences: list[str]

@define
class InverseFoldingConfig:
    invalid_ids: Sequence[int] = []
    temperature: float = 1.0

@define
class SamplingTrackConfig:
    temperature: float = 1.0
    top_p: float = 1.0
    only_sample_masked_tokens: bool = True
    invalid_ids: Sequence[int] = []
    topk_logprobs: int = 0

@define
class SamplingConfig:
    sequence: SamplingTrackConfig | None = attr.field(default=None, metadata={'max_topk': C.MAX_TOPK_SEQUENCE})
    structure: SamplingTrackConfig | None = attr.field(default=None, metadata={'max_topk': C.MAX_TOPK_STRUCTURE})
    secondary_structure: SamplingTrackConfig | None = attr.field(default=None, metadata={'max_topk': C.MAX_TOPK_SECONDARY_STRUCTURE})
    sasa: SamplingTrackConfig | None = attr.field(default=None, metadata={'max_topk': C.MAX_TOPK_SASA})
    function: SamplingTrackConfig | None = attr.field(default=None, metadata={'max_topk': C.MAX_TOPK_FUNCTION})
    return_per_residue_embeddings: bool = False
    return_mean_embedding: bool = False

@define
class ForwardTrackData:
    sequence: torch.Tensor | None = None
    structure: torch.Tensor | None = None
    secondary_structure: torch.Tensor | None = None
    sasa: torch.Tensor | None = None
    function: torch.Tensor | None = None

@define
class LogitsConfig:
    sequence: bool = False
    structure: bool = False
    secondary_structure: bool = False
    sasa: bool = False
    function: bool = False
    residue_annotations: bool = False
    return_embeddings: bool = False
    return_hidden_states: bool = False
    ith_hidden_layer: int = -1

@define
class LogitsOutput:
    logits: ForwardTrackData | None = None
    embeddings: torch.Tensor | None = None
    residue_annotation_logits: torch.Tensor | None = None
    hidden_states: torch.Tensor | None = None

@define
class ForwardAndSampleOutput(LogitsOutput):
    protein_tensor: ESMProteinTensor = ESMProteinTensor()
    entropy: ForwardTrackData | None = None
    prob: ForwardTrackData | None = None
    logprob: ForwardTrackData | None = None
    top_prob: ForwardTrackData | None = None
    topk_logprob: ForwardTrackData | None = None
    topk_tokens: ForwardTrackData | None = None
    per_residue_embedding: torch.Tensor | None = None
    mean_embedding: torch.Tensor | None = None

class ESM3InferenceClient(ABC):

    def generate(self, input: ProteinType, config: GenerationConfig) -> ProteinType:...

    def batch_generate(self, inputs: Sequence[ProteinType], configs: Sequence[GenerationConfig]) -> ...:...

    def encode(self, input: ESMProtein) -> ESMProteinTensor:...

    def decode(self, input: ESMProteinTensor) -> ESMProtein:...

    def logits(self, input: ESMProteinTensor, config: LogitsConfig=LogitsConfig()) -> ...:...

    def forward_and_sample(self, input: ESMProteinTensor, sampling_configuration: SamplingConfig) -> ...:...

    @property
    def raw_model(self):...

class ESMCInferenceClient(ABC):

    def encode(self, input: ESMProtein) -> ESMProteinTensor:...

    def decode(self, input: ESMProteinTensor) -> ESMProtein:...

    def logits(self, input: ESMProteinTensor, config: LogitsConfig=LogitsConfig()) -> ...:...

    @property
    def raw_model(self):...
```

`/esm/esm/utils/structure/aligner.py`:

```python
from __future__ import annotations
from dataclasses import Field, replace
from typing import Any, ClassVar, Protocol, TypeVar
import numpy as np
import torch
from esm.utils.structure.protein_structure import compute_affine_and_rmsd

class Alignable(Protocol):
    atom37_positions: np.ndarray
    atom37_mask: np.ndarray
    __dataclass_fields__: ClassVar[dict[str, Field[Any]]]

    def __len__(self) -> int:
        ...
T = TypeVar('T', bound=Alignable)

class Aligner:

    def __init__(self, mobile: Alignable, target: Alignable, only_use_backbone: bool=False, use_reflection: bool=False):...

    @property
    def rmsd(self):...

    def apply(self, mobile: T) -> T:...
```

`/esm/esm/pretrained.py`:

```python
from typing import Callable
import torch
import torch.nn as nn
from esm.models.esm3 import ESM3
from esm.models.esmc import ESMC
from esm.models.function_decoder import FunctionTokenDecoder
from esm.models.vqvae import StructureTokenDecoder, StructureTokenEncoder
from esm.tokenization import get_esm3_model_tokenizers, get_esmc_model_tokenizers
from esm.utils.constants.esm3 import data_root
from esm.utils.constants.models import ESM3_FUNCTION_DECODER_V0, ESM3_OPEN_SMALL, ESM3_STRUCTURE_DECODER_V0, ESM3_STRUCTURE_ENCODER_V0, ESMC_300M, ESMC_600M
ModelBuilder = Callable[[torch.device | str], nn.Module]

def ESM3_structure_encoder_v0(device: torch.device | str='cpu'):
    with torch.device(device):
        model = StructureTokenEncoder(d_model=1024, n_heads=1, v_heads=128, n_layers=2, d_out=128, n_codes=4096).eval()
    state_dict = torch.load(data_root('esm3') / 'data/weights/esm3_structure_encoder_v0.pth', map_location=device)
    model.load_state_dict(state_dict)
    return model

def ESM3_structure_decoder_v0(device: torch.device | str='cpu'):
    with torch.device(device):
        model = StructureTokenDecoder(d_model=1280, n_heads=20, n_layers=30).eval()
    state_dict = torch.load(data_root('esm3') / 'data/weights/esm3_structure_decoder_v0.pth', map_location=device)
    model.load_state_dict(state_dict)
    return model

def ESM3_function_decoder_v0(device: torch.device | str='cpu'):
    with torch.device(device):
        model = FunctionTokenDecoder().eval()
    state_dict = torch.load(data_root('esm3') / 'data/weights/esm3_function_decoder_v0.pth', map_location=device)
    model.load_state_dict(state_dict)
    return model

def ESMC_300M_202412(device: torch.device | str='cpu', use_flash_attn: bool=True):...

def ESMC_600M_202412(device: torch.device | str='cpu', use_flash_attn: bool=True):...

def ESM3_sm_open_v0(device: torch.device | str='cpu'):
    with torch.device(device):
        model = ESM3(d_model=1536, n_heads=24, v_heads=256, n_layers=48, structure_encoder_fn=ESM3_structure_encoder_v0, structure_decoder_fn=ESM3_structure_decoder_v0, function_decoder_fn=ESM3_function_decoder_v0, tokenizers=get_esm3_model_tokenizers(ESM3_OPEN_SMALL)).eval()
    state_dict = torch.load(data_root('esm3') / 'data/weights/esm3_sm_open_v1.pth', map_location=device)
    model.load_state_dict(state_dict)
    return model
LOCAL_MODEL_REGISTRY: dict[str, ModelBuilder] = {ESM3_OPEN_SMALL: ESM3_sm_open_v0, ESM3_STRUCTURE_ENCODER_V0: ESM3_structure_encoder_v0, ESM3_STRUCTURE_DECODER_V0: ESM3_structure_decoder_v0, ESM3_FUNCTION_DECODER_V0: ESM3_function_decoder_v0, ESMC_600M: ESMC_600M_202412, ESMC_300M: ESMC_300M_202412}

def load_local_model(model_name: str, device: torch.device=torch.device('cpu')) -> nn.Module:
    if model_name not in LOCAL_MODEL_REGISTRY:...
    return LOCAL_MODEL_REGISTRY[model_name](device)

def register_local_model(model_name: str, model_builder: ModelBuilder) -> None:...
```

`/esm/esm/utils/constants/models.py`:

```python
ESM3_OPEN_SMALL = 'esm3_sm_open_v1'
ESM3_OPEN_SMALL_ALIAS_1 = 'esm3-open-2024-03'
ESM3_OPEN_SMALL_ALIAS_2 = 'esm3-sm-open-v1'
ESM3_OPEN_SMALL_ALIAS_3 = 'esm3-open'
ESM3_STRUCTURE_ENCODER_V0 = 'esm3_structure_encoder_v0'
ESM3_STRUCTURE_DECODER_V0 = 'esm3_structure_decoder_v0'
ESM3_FUNCTION_DECODER_V0 = 'esm3_function_decoder_v0'
ESMC_600M = 'esmc_600m'
ESMC_300M = 'esmc_300m'

def forge_only_return_single_layer_hidden_states(model_name: str):...

def model_is_locally_supported(x: str):...

def normalize_model_name(x: str):
    if x in {ESM3_OPEN_SMALL_ALIAS_1, ESM3_OPEN_SMALL_ALIAS_2, ESM3_OPEN_SMALL_ALIAS_3}:
        return ESM3_OPEN_SMALL
    return x
```

`/esm/esm/utils/misc.py`:

```python
import os
from collections import defaultdict
from io import BytesIO
from typing import Any, ContextManager, Sequence, TypeVar
from warnings import warn
import huggingface_hub
import numpy as np
import torch
import zstd
from esm.utils.constants.esm3 import CHAIN_BREAK_STR
from esm.utils.types import FunctionAnnotation
MAX_SUPPORTED_DISTANCE = 1000000.0
TSequence = TypeVar('TSequence', bound=Sequence)

def slice_python_object_as_numpy(obj: TSequence, idx: int | list[int] | slice | np.ndarray) -> TSequence:...
    if isinstance(idx, int):...
    if isinstance(idx, np.ndarray) and idx.dtype == bool:
        sliced_obj = [obj[i] for i in np.where(idx)[0]]
    elif isinstance(idx, slice):
        sliced_obj = obj[idx]
    else:...
    match (obj, sliced_obj):
        case [str(), list()]:
            sliced_obj = ''.join(sliced_obj)
        case _:
            sliced_obj = obj.__class__(sliced_obj)
    return sliced_obj

def rbf(values, v_min, v_max, n_bins=16):...
    rbf_centers = torch.linspace(v_min, v_max, n_bins, device=values.device, dtype=values.dtype)
    rbf_centers = rbf_centers.view([1] * len(values.shape) + [-1])
    rbf_std = (v_max - v_min) / n_bins
    z = (values.unsqueeze(-1) - rbf_centers) / rbf_std
    return torch.exp(-z ** 2)

def batched_gather(data, inds, dim=0, no_batch_dims=0):...

def node_gather(s: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:...

def knn_graph(coords: torch.Tensor, coord_mask: torch.Tensor, padding_mask: torch.Tensor, sequence_id: torch.Tensor, *, no_knn: int):
    L = coords.shape[-2]
    num_by_dist = min(no_knn, L)
    device = coords.device
    coords = coords.nan_to_num()
    coord_mask = ~(coord_mask[..., None, :] & coord_mask[..., :, None])
    padding_pairwise_mask = padding_mask[..., None, :] | padding_mask[..., :, None]
    if sequence_id is not None:
        padding_pairwise_mask |= torch.unsqueeze(sequence_id, 1) != torch.unsqueeze(sequence_id, 2)
    dists = (coords.unsqueeze(-2) - coords.unsqueeze(-3)).norm(dim=-1)
    arange = torch.arange(L, device=device)
    seq_dists = (arange.unsqueeze(-1) - arange.unsqueeze(-2)).abs()
    max_dist = MAX_SUPPORTED_DISTANCE
    torch._assert_async((dists[~coord_mask] < max_dist).all())
    struct_then_seq_dist = seq_dists.to(dists.dtype).mul(100.0).add(max_dist).where(coord_mask, dists).masked_fill(padding_pairwise_mask, torch.inf)
    (dists, edges) = struct_then_seq_dist.sort(dim=-1, descending=False)
    chosen_edges = edges[..., :num_by_dist]
    chosen_mask = dists[..., :num_by_dist].isfinite()
    return (chosen_edges, chosen_mask)

def stack_variable_length_tensors(sequences: Sequence[torch.Tensor], constant_value: int | float=0, dtype: torch.dtype | None=None) -> torch.Tensor:...
    batch_size = len(sequences)
    shape = [batch_size] + np.max([seq.shape for seq in sequences], 0).tolist()
    if dtype is None:
        dtype = sequences[0].dtype
    device = sequences[0].device
    array = torch.full(shape, constant_value, dtype=dtype, device=device)
    for (arr, seq) in zip(array, sequences):
        arrslice = tuple((slice(dim) for dim in seq.shape))
        arr[arrslice] = seq
    return array

def unbinpack(tensor: torch.Tensor, sequence_id: torch.Tensor | None, pad_value: int | float):...

def fp32_autocast_context(device_type: str) -> ContextManager[torch.amp.autocast]:...
    if device_type == 'cpu':
        return torch.amp.autocast(device_type, enabled=False)
    elif device_type == 'cuda':
        return torch.amp.autocast(device_type, dtype=torch.float32)
    else:...

def merge_ranges(ranges: list[range], merge_gap_max: int | None=None) -> list[range]:...

def merge_annotations(annotations: list[FunctionAnnotation], merge_gap_max: int | None=None) -> list[FunctionAnnotation]:...

def replace_inf(data):...

def maybe_tensor(x, convert_none_to_nan: bool=False) -> torch.Tensor | None:...

def maybe_list(x, convert_nan_to_none: bool=False) -> list | None:...

def huggingfacehub_login():...

def get_chainbreak_boundaries_from_sequence(sequence: Sequence[str]) -> np.ndarray:
    chain_boundaries = [0]
    for (i, aa) in enumerate(sequence):
        if aa == CHAIN_BREAK_STR:...
    chain_boundaries.append(len(sequence))
    assert len(chain_boundaries) % 2 == 0
    chain_boundaries = np.array(chain_boundaries).reshape(-1, 2)
    return chain_boundaries

def deserialize_tensors(b: bytes) -> Any:...
```

`/esm/esm/models/function_decoder.py`:

```python
from collections import defaultdict
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from cloudpathlib import AnyPath
from esm.layers.regression_head import RegressionHead
from esm.layers.transformer_stack import TransformerStack
from esm.tokenization.function_tokenizer import InterProQuantizedTokenizer
from esm.utils.constants import esm3 as C
from esm.utils.misc import merge_annotations, merge_ranges
from esm.utils.types import FunctionAnnotation

@dataclass(frozen=True)
class FunctionTokenDecoderConfig:
    d_model: int = 1024
    n_heads: int = 8
    n_layers: int = 3
    function_token_vocab_size: int = 260
    function_token_depth: int = 8
    num_interpro_classes: int = 29026
    keyword_vocabulary_size: int = 58641
    interpro_entry_list: str = field(default_factory=lambda : str(C.INTERPRO_ENTRY))
    keyword_vocabulary_path: str = field(default_factory=lambda : str(C.data_root('esm3') / C.KEYWORDS_VOCABULARY))
    unpack_lsh_bits: bool = True
    num_special_tokens: int = 4
    bits_per_token: int = 8

class FunctionTokenDecoder(nn.Module):

    def __init__(self, config: FunctionTokenDecoderConfig | None=None):...
        super().__init__()
        if config is None:
            config = FunctionTokenDecoderConfig()
        self.config = config
        with AnyPath(config.interpro_entry_list).open('r') as f:
            df = pd.read_csv(f, sep='\t')
        self.interpro_ids = sorted(df.ENTRY_AC)
        self.interpro2index = {interpro_id: i for (i, interpro_id) in enumerate(self.interpro_ids)}
        assert len(self.interpro_ids) == config.num_interpro_classes
        with AnyPath(config.keyword_vocabulary_path).open('r') as f:
            self.keywords_vocabulary: list[str] = list(f.read().strip().split('\n'))
            assert len(self.keywords_vocabulary) == config.keyword_vocabulary_size
        if config.unpack_lsh_bits:
            vocab_size = 2 * config.function_token_depth * config.bits_per_token
        else:...
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=config.d_model)
        self.decoder = TransformerStack(d_model=config.d_model, n_heads=config.n_heads, v_heads=None, n_layers=config.n_layers, n_layers_geom=0, scale_residue=False, bias=True, qk_layernorm=False, ffn_type='gelu', expansion_ratio=4)
        self.heads = nn.ModuleDict({'keyword_logits': RegressionHead(d_model=config.d_model, output_dim=config.keyword_vocabulary_size, hidden_dim=4 * config.d_model), 'keyword_tfidf': RegressionHead(d_model=config.d_model, output_dim=config.keyword_vocabulary_size, hidden_dim=4 * config.d_model), 'interpro_logits': RegressionHead(d_model=config.d_model, output_dim=config.num_interpro_classes, hidden_dim=4 * config.d_model)})

    def forward(self, token_ids: torch.Tensor) -> dict[str, torch.Tensor]:...

    @property
    def device(self) -> torch.device:...

    def decode(self, function_token_ids: torch.Tensor, tokenizer: InterProQuantizedTokenizer, decode_annotations: bool=True, annotation_threshold: float=0.1, decode_keywords=True, keywords_threshold: float=0.5, annotation_min_length: int | None=5, annotation_gap_merge_max: int | None=3):...

    def _preds_to_keywords(self, keyword_preds: np.ndarray) -> list[FunctionAnnotation]:...
```

`/esm/esm/models/vqvae.py`:

```python
import torch
import torch.nn as nn
from esm.layers.blocks import UnifiedTransformerBlock
from esm.layers.codebook import EMACodebook
from esm.layers.structure_proj import Dim6RotStructureHead
from esm.layers.transformer_stack import TransformerStack
from esm.utils.constants import esm3 as C
from esm.utils.misc import knn_graph
from esm.utils.structure.affine3d import Affine3D, build_affine3d_from_coordinates
from esm.utils.structure.predicted_aligned_error import compute_predicted_aligned_error, compute_tm

class RelativePositionEmbedding(nn.Module):

    def __init__(self, bins, embedding_dim, init_std=0.02):
        super().__init__()
        self.bins = bins
        self.embedding = torch.nn.Embedding(2 * bins + 2, embedding_dim)
        self.embedding.weight.data.normal_(0, init_std)

    def forward(self, query_residue_index, key_residue_index):...
        assert query_residue_index.dtype == torch.long
        assert key_residue_index.dtype == torch.long
        assert query_residue_index.ndim == 1
        assert key_residue_index.ndim == 2
        diff = key_residue_index - query_residue_index.unsqueeze(1)
        diff = diff.clamp(-self.bins, self.bins)
        diff = diff + self.bins + 1
        output = self.embedding(diff)
        return output

class PairwisePredictionHead(nn.Module):

    def __init__(self, input_dim: int, downproject_dim: int, hidden_dim: int, n_bins: int, bias: bool=True, pairwise_state_dim: int=0):
        super().__init__()
        self.downproject = nn.Linear(input_dim, downproject_dim, bias=bias)
        self.linear1 = nn.Linear(downproject_dim + pairwise_state_dim, hidden_dim, bias=bias)
        self.activation_fn = nn.GELU()
        self.norm = nn.LayerNorm(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, n_bins, bias=bias)

    def forward(self, x, pairwise: torch.Tensor | None=None):...
        x = self.downproject(x)
        (q, k) = x.chunk(2, dim=-1)
        prod = q[:, None, :, :] * k[:, :, None, :]
        diff = q[:, None, :, :] - k[:, :, None, :]
        x_2d = [prod, diff]
        if pairwise is not None:...
        x = torch.cat(x_2d, dim=-1)
        x = self.linear1(x)
        x = self.activation_fn(x)
        x = self.norm(x)
        x = self.linear2(x)
        return x

class RegressionHead(nn.Module):

    def __init__(self, embed_dim: int, output_dim: int):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.activation_fn = nn.GELU()
        self.norm = nn.LayerNorm(embed_dim)
        self.output = nn.Linear(embed_dim, output_dim)

    def forward(self, features):
        x = self.dense(features)
        x = self.activation_fn(x)
        x = self.norm(x)
        x = self.output(x)
        return x

class CategoricalMixture:

    def __init__(self, param, bins=50, start=0, end=1):
        self.logits = param
        bins = torch.linspace(start, end, bins + 1, device=self.logits.device, dtype=torch.float32)
        self.v_bins = (bins[:-1] + bins[1:]) / 2

    def log_prob(self, true):...

    def mean(self):
        return (self.logits.to(self.v_bins.dtype).softmax(-1) @ self.v_bins.unsqueeze(1)).squeeze(-1)

    def median(self):...

class GeometricEncoderStack(TransformerStack):

    def __init__(self, d_model, n_heads, v_heads, n_layers):
        super().__init__(d_model, n_heads, v_heads, 0)
        self.blocks = nn.ModuleList([UnifiedTransformerBlock(d_model, n_heads, v_heads=v_heads, use_geom_attn=True, use_plain_attn=False, expansion_ratio=4, bias=True) for i in range(n_layers)])
        self.norm = nn.Identity()

def batched_gather(data, inds, dim=0, no_batch_dims=0):
    ranges = []
    for (i, s) in enumerate(data.shape[:no_batch_dims]):
        r = torch.arange(s)
        r = r.view(*(*(1,) * i, -1, *(1,) * (len(inds.shape) - i - 1)))
        ranges.append(r)
    remaining_dims = [slice(None) for _ in range(len(data.shape) - no_batch_dims)]
    remaining_dims[dim - no_batch_dims if dim >= 0 else dim] = inds
    ranges.extend(remaining_dims)
    return data[ranges]

def node_gather(s: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    return batched_gather(s.unsqueeze(-3), edges, -2, no_batch_dims=len(s.shape) - 1)

class StructureTokenEncoder(nn.Module):

    def __init__(self, d_model, n_heads, v_heads, n_layers, d_out, n_codes):
        super().__init__()
        self.transformer = GeometricEncoderStack(d_model, n_heads, v_heads, n_layers)
        self.pre_vq_proj = nn.Linear(d_model, d_out)
        self.codebook = EMACodebook(n_codes, d_out)
        self.relative_positional_embedding = RelativePositionEmbedding(32, d_model, init_std=0.02)
        self.knn = 16

    def encode_local_structure(self, coords: torch.Tensor, affine: Affine3D, attention_mask: torch.Tensor, sequence_id: torch.Tensor | None, affine_mask: torch.Tensor, residue_index: torch.Tensor | None=None):...
        assert coords.size(-1) == 3 and coords.size(-2) == 3, 'need N, CA, C'
        with torch.no_grad():
            (knn_edges, _) = self.find_knn_edges(coords, ~attention_mask, coord_mask=affine_mask, sequence_id=sequence_id, knn=self.knn)
            (B, L, E) = knn_edges.shape
            affine_tensor = affine.tensor
            T_D = affine_tensor.size(-1)
            knn_affine_tensor = node_gather(affine_tensor, knn_edges)
            knn_affine_tensor = knn_affine_tensor.view(-1, E, T_D).contiguous()
            affine = Affine3D.from_tensor(knn_affine_tensor)
            knn_sequence_id = node_gather(sequence_id.unsqueeze(-1), knn_edges).view(-1, E) if sequence_id is not None else ...
            knn_affine_mask = node_gather(affine_mask.unsqueeze(-1), knn_edges).view(-1, E)
            knn_chain_id = torch.zeros(B * L, E, dtype=torch.int64, device=coords.device)
            if residue_index is None:...
            else:
                res_idxs = node_gather(residue_index.unsqueeze(-1), knn_edges).view(-1, E)
        z = self.relative_positional_embedding(res_idxs[:, 0], res_idxs)
        (z, _, _) = self.transformer.forward(x=z, sequence_id=knn_sequence_id, affine=affine, affine_mask=knn_affine_mask, chain_id=knn_chain_id)
        z = z.view(B, L, E, -1)
        z = z[:, :, 0, :]
        return z

    @staticmethod
    def find_knn_edges(coords, padding_mask, coord_mask, sequence_id: torch.Tensor | None=None, knn: int | None=None) -> tuple:
        assert knn is not None, 'Must specify a non-null knn to find_knn_edges'
        coords = coords.clone()
        coords[~coord_mask] = 0
        if sequence_id is None:...
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
            ca = coords[..., 1, :]
            (edges, edge_mask) = knn_graph(ca, coord_mask, padding_mask, sequence_id, no_knn=knn)
        return (edges, edge_mask)

    def encode(self, coords: torch.Tensor, attention_mask: torch.Tensor | None=None, sequence_id: torch.Tensor | None=None, residue_index: torch.Tensor | None=None):
        coords = coords[..., :3, :]
        (affine, affine_mask) = build_affine3d_from_coordinates(coords=coords)
        if attention_mask is None:
            attention_mask = torch.ones_like(affine_mask, dtype=torch.bool)
        attention_mask = attention_mask.bool()
        if sequence_id is None:
            sequence_id = torch.zeros_like(affine_mask, dtype=torch.int64)
        z = self.encode_local_structure(coords=coords, affine=affine, attention_mask=attention_mask, sequence_id=sequence_id, affine_mask=affine_mask, residue_index=residue_index)
        z = z.masked_fill(~affine_mask.unsqueeze(2), 0)
        z = self.pre_vq_proj(z)
        (z_q, min_encoding_indices, _) = self.codebook(z)
        return (z_q, min_encoding_indices)

class StructureTokenDecoder(nn.Module):

    def __init__(self, d_model, n_heads, n_layers):
        super().__init__()
        self.decoder_channels = d_model
        self.vqvae_codebook_size = C.VQVAE_CODEBOOK_SIZE
        self.special_tokens = C.VQVAE_SPECIAL_TOKENS
        self.max_pae_bin = C.VQVAE_MAX_PAE_BIN
        self.embed = nn.Embedding(self.vqvae_codebook_size + len(self.special_tokens), d_model)
        self.decoder_stack = TransformerStack(d_model, n_heads, 1, n_layers, scale_residue=False, n_layers_geom=0)
        self.affine_output_projection = Dim6RotStructureHead(self.decoder_channels, 10, predict_torsion_angles=False)
        direction_loss_bins = C.VQVAE_DIRECTION_LOSS_BINS
        pae_bins = C.VQVAE_PAE_BINS
        self.pairwise_bins = [64, direction_loss_bins * 6, pae_bins]
        self.pairwise_classification_head = PairwisePredictionHead(self.decoder_channels, downproject_dim=128, hidden_dim=128, n_bins=sum(self.pairwise_bins), bias=False)
        plddt_bins = C.VQVAE_PLDDT_BINS
        self.plddt_head = RegressionHead(embed_dim=self.decoder_channels, output_dim=plddt_bins)

    def decode(self, structure_tokens: torch.Tensor, attention_mask: torch.Tensor | None=None, sequence_id: torch.Tensor | None=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(structure_tokens, dtype=torch.bool)
        attention_mask = attention_mask.bool()
        if sequence_id is None:
            sequence_id = torch.zeros_like(structure_tokens, dtype=torch.int64)
        chain_id = torch.zeros_like(structure_tokens, dtype=torch.int64)
        assert structure_tokens[:, 0].eq(self.special_tokens['BOS']).all(), ...
        assert structure_tokens[torch.arange(structure_tokens.shape[0]), attention_mask.sum(1) - 1].eq(self.special_tokens['EOS']).all(), ...
        assert (structure_tokens < 0).sum() == 0, ...
        x = self.embed(structure_tokens)
        (x, _, _) = self.decoder_stack.forward(x, affine=None, affine_mask=None, sequence_id=sequence_id, chain_id=chain_id)
        (tensor7_affine, bb_pred) = self.affine_output_projection(x, affine=None, affine_mask=torch.zeros_like(attention_mask))
        (pae, ptm) = (None, None)
        pairwise_logits = self.pairwise_classification_head(x)
        (_, _, pae_logits) = [o if o.numel() > 0 else None for o in pairwise_logits.split(self.pairwise_bins, dim=-1)]
        special_tokens_mask = structure_tokens >= min(self.special_tokens.values())
        pae = compute_predicted_aligned_error(pae_logits, aa_mask=~special_tokens_mask, sequence_id=sequence_id, max_bin=self.max_pae_bin)
        ptm = compute_tm(pae_logits, aa_mask=~special_tokens_mask, max_bin=self.max_pae_bin)
        plddt_logits = self.plddt_head(x)
        plddt_value = CategoricalMixture(plddt_logits, bins=plddt_logits.shape[-1]).mean()
        return dict(tensor7_affine=tensor7_affine, bb_pred=bb_pred, plddt=plddt_value, ptm=ptm, predicted_aligned_error=pae)
```

`/esm/esm/utils/structure/protein_structure.py`:

```python
from __future__ import annotations
from typing import Tuple, TypeVar
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.amp import autocast
from esm.utils import residue_constants
from esm.utils.misc import unbinpack
from esm.utils.structure.affine3d import Affine3D
ArrayOrTensor = TypeVar('ArrayOrTensor', np.ndarray, Tensor)

def index_by_atom_name(atom37: ArrayOrTensor, atom_names: str | list[str], dim: int=-2) -> ...:
    squeeze = False
    if isinstance(atom_names, str):...
    indices = [residue_constants.atom_order[atom_name] for atom_name in atom_names]
    dim = dim % atom37.ndim
    index = tuple((slice(None) if dim != i else indices for i in range(atom37.ndim)))
    result = atom37[index]
    if squeeze:...
    return result

def infer_cbeta_from_atom37(atom37: ArrayOrTensor, L: float=1.522, A: float=1.927, D: float=-2.143):...

@torch.no_grad()
@autocast('cuda', enabled=False)
def compute_alignment_tensors(mobile: torch.Tensor, target: torch.Tensor, atom_exists_mask: torch.Tensor | None=None, sequence_id: torch.Tensor | None=None):...

@torch.no_grad()
@autocast('cuda', enabled=False)
def compute_rmsd_no_alignment(aligned: torch.Tensor, target: torch.Tensor, num_valid_atoms: torch.Tensor, reduction: str='batch') -> ...:...

@torch.no_grad()
@autocast('cuda', enabled=False)
def compute_affine_and_rmsd(mobile: torch.Tensor, target: torch.Tensor, atom_exists_mask: torch.Tensor | None=None, sequence_id: torch.Tensor | None=None) -> ...:...

def compute_gdt_ts_no_alignment(aligned: torch.Tensor, target: torch.Tensor, atom_exists_mask: torch.Tensor, reduction: str='batch') -> ...:...
```

`/esm/esm/utils/forge_context_manager.py`:

```python
import threading
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextvars import copy_context
from typing import Any, Callable, Dict, List
from tqdm import tqdm
from esm.sdk.api import ESMProteinError
from esm.sdk.forge import retry_if_specific_error, skip_retries_var
TQDM_BAR_FORMAT = '{desc:<12}{percentage:3.0f}%|{bar:24}| {n_fmt}/{total_fmt} [Elapsed: {elapsed} | Remaining: {remaining}] {postfix}'

class AIMDRateLimiter:

    def __init__(self, initial_concurrency: int=32, min_concurrency: int=1, max_concurrency: int=512, step_up: int=1):...

    def adjust_concurrency(self, error_seen: bool) -> int:...

class ForgeBatchExecutor:

    def __init__(self, max_attempts: int=10):...

    def __enter__(self):...

    def __exit__(self, exc_type, exc_val, exc_tb):...

    def _validate_inputs(self, inputs: Dict[str, Any]) -> int:...

    def execute_batch(self, user_func: Callable, **kwargs: Any) -> List[Any]:...
```

`/esm/esm/layers/rotary.py`:

```python
from typing import Tuple
import torch
from einops import rearrange, repeat
try:
    from flash_attn.ops.triton.rotary import apply_rotary as apply_triton_rotary
except ImportError:
    apply_triton_rotary = None

def rotate_half(x, interleaved=False):
    if not interleaved:
        (x1, x2) = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    else:...

def apply_rotary_emb_torch(x, cos, sin, interleaved=False, _inplace=False):...
    ro_dim = cos.shape[-1] * 2
    assert ro_dim <= x.shape[-1]
    seqlen = x.size(1)
    cos = cos[:seqlen]
    sin = sin[:seqlen]
    cos = repeat(cos, 's d -> s 1 (2 d)')
    sin = repeat(sin, 's d -> s 1 (2 d)')
    return torch.cat([x[..., :ro_dim] * cos + rotate_half(x[..., :ro_dim], interleaved) * sin, x[..., ro_dim:]], dim=-1)

class RotaryEmbedding(torch.nn.Module):

    def __init__(self, dim: int, base=10000.0, interleaved=False, scale_base=None, scaling_factor=1.0, pos_idx_in_fp32=True, device=None):...
        super().__init__()
        self.dim = dim
        self.base = float(base)
        self.pos_idx_in_fp32 = pos_idx_in_fp32
        self.interleaved = interleaved
        self.scale_base = scale_base
        self.scaling_factor = scaling_factor
        self.device = device
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
        self._cos_k_cached = None
        self._sin_k_cached = None
        self.reset_parameters()

    def reset_parameters(self):
        inv_freq = self._compute_inv_freq(self.device)
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        arange = torch.arange(0, self.dim, 2, device=self.device, dtype=torch.float32)
        scale = ... if self.scale_base is not None else None
        self.register_buffer('scale', scale)

    def _compute_inv_freq(self, device=None):
        return 1 / self.base ** (torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim)

    def _update_cos_sin_cache(self, seqlen, device=None, dtype=None):
        if seqlen > self._seq_len_cached or self._cos_cached is None or self._cos_cached.device != device or (self._cos_cached.dtype != dtype) or (self.training and self._cos_cached.is_inference()):
            self._seq_len_cached = seqlen
            if self.pos_idx_in_fp32:
                t = torch.arange(seqlen, device=device, dtype=torch.float32)
                t /= self.scaling_factor
                if self.inv_freq.dtype != torch.float32:
                    inv_freq = self.inv_freq.to(torch.float32)
                else:
                    inv_freq = self.inv_freq
            else:...
            freqs = torch.outer(t, inv_freq)
            if self.scale is None:
                self._cos_cached = torch.cos(freqs).to(dtype)
                self._sin_cached = torch.sin(freqs).to(dtype)
            else:...

    def forward(self, q: torch.Tensor, k: torch.Tensor, seqlen_offset: int=0) -> Tuple[torch.Tensor, torch.Tensor]:...
        self._update_cos_sin_cache(q.shape[1] + seqlen_offset, device=q.device, dtype=q.dtype)
        assert self._cos_cached is not None
        assert self._sin_cached is not None
        if self.scale is None:
            return (apply_rotary_emb_torch(q, self._cos_cached[seqlen_offset:], self._sin_cached[seqlen_offset:], self.interleaved, True), apply_rotary_emb_torch(k, self._cos_cached[seqlen_offset:], self._sin_cached[seqlen_offset:], self.interleaved, True))
        else:...

class TritonRotaryEmbedding(RotaryEmbedding):

    def forward(self, qkv: torch.Tensor, cu_seqlens, max_seqlen) -> torch.Tensor:...
```

`/esm/esm/utils/noise_schedules.py`:

```python
import math
import torch

def cosine_schedule(t: torch.Tensor):
    return torch.cos(t * math.pi * 0.5)

def cubic_schedule(t):...

def linear_schedule(t):...

def square_root_schedule(t):...

def square_schedule(t):...
NOISE_SCHEDULE_REGISTRY = {'cosine': cosine_schedule, 'linear': linear_schedule, 'square_root_schedule': square_root_schedule, 'cubic': cubic_schedule, 'square': square_schedule}
```

`/esm/esm/models/esm3.py`:

```python
from __future__ import annotations
import contextlib
from functools import partial
from typing import Callable
import attr
import einops
import torch
import torch.nn as nn
from attr import dataclass
from esm.layers.regression_head import RegressionHead
from esm.layers.transformer_stack import TransformerStack
from esm.models.function_decoder import FunctionTokenDecoder
from esm.models.vqvae import StructureTokenDecoder, StructureTokenEncoder
from esm.sdk.api import ESM3InferenceClient, ESMProtein, ESMProteinTensor, ForwardAndSampleOutput, ForwardTrackData, GenerationConfig, LogitsConfig, LogitsOutput, ProteinType, SamplingConfig
from esm.tokenization import TokenizerCollectionProtocol
from esm.utils import encoding
from esm.utils.constants import esm3 as C
from esm.utils.constants.models import ESM3_OPEN_SMALL, normalize_model_name
from esm.utils.decoding import decode_protein_tensor
from esm.utils.generation import _batch_forward, _sample_per_prompt, _slice_tensor_dataclass, iterative_sampling_raw, iterative_sampling_tokens
from esm.utils.misc import rbf
from esm.utils.sampling import _BatchedESMProteinTensor, get_default_sampling_config, validate_sampling_config
from esm.utils.structure.affine3d import build_affine3d_from_coordinates

@dataclass
class ESMOutput:
    sequence_logits: torch.Tensor
    structure_logits: torch.Tensor
    secondary_structure_logits: torch.Tensor
    sasa_logits: torch.Tensor
    function_logits: torch.Tensor
    residue_logits: torch.Tensor
    embeddings: torch.Tensor

class EncodeInputs(nn.Module):

    def __init__(self, d_model: int):
        super().__init__()
        self.sequence_embed = nn.Embedding(64, d_model)
        self.plddt_projection = nn.Linear(16, d_model)
        self.structure_per_res_plddt_projection = nn.Linear(16, d_model)
        self.structure_tokens_embed = nn.Embedding(4096 + 5, d_model)
        self.ss8_embed = nn.Embedding(8 + 3, d_model)
        self.sasa_embed = nn.Embedding(16 + 3, d_model)
        self.function_embed = nn.ModuleList([nn.Embedding(260, d_model // 8, padding_idx=0) for _ in range(8)])
        self.residue_embed = nn.EmbeddingBag(1478, d_model, mode='sum', padding_idx=0)

    def forward(self, sequence_tokens: torch.Tensor, structure_tokens: torch.Tensor, average_plddt: torch.Tensor, per_res_plddt: torch.Tensor, ss8_tokens: torch.Tensor, sasa_tokens: torch.Tensor, function_tokens: torch.Tensor, residue_annotation_tokens: torch.Tensor) -> ...:
        sequence_embed = self.sequence_embed(sequence_tokens)
        rbf_16_fn = partial(rbf, v_min=0.0, v_max=1.0, n_bins=16)
        plddt_embed = self.plddt_projection(rbf_16_fn(average_plddt))
        structure_per_res_plddt = self.structure_per_res_plddt_projection(rbf_16_fn(per_res_plddt))
        structure_embed = self.structure_tokens_embed(structure_tokens)
        ss8_embed = self.ss8_embed(ss8_tokens)
        sasa_embed = self.sasa_embed(sasa_tokens)
        function_embed = torch.cat([embed_fn(funcs) for (embed_fn, funcs) in zip(self.function_embed, function_tokens.unbind(-1))], -1)
        (B, L, N) = residue_annotation_tokens.shape
        residue_embed = self.residue_embed(einops.rearrange(residue_annotation_tokens, 'B L N -> (B L) N', B=B, L=L, N=N))
        residue_embed = einops.rearrange(residue_embed, '(B L) D -> B L D', B=B, L=L)
        return sequence_embed + plddt_embed + structure_per_res_plddt + structure_embed + ss8_embed + sasa_embed + function_embed + residue_embed

class OutputHeads(nn.Module):

    def __init__(self, d_model: int):
        super().__init__()
        self.sequence_head = RegressionHead(d_model, 64)
        self.structure_head = RegressionHead(d_model, 4096)
        self.ss8_head = RegressionHead(d_model, 8 + 3)
        self.sasa_head = RegressionHead(d_model, 16 + 3)
        self.function_head = RegressionHead(d_model, 260 * 8)
        self.residue_head = RegressionHead(d_model, 1478)

    def forward(self, x: torch.Tensor, embed: torch.Tensor) -> ESMOutput:
        sequence_logits = self.sequence_head(x)
        structure_logits = self.structure_head(x)
        secondary_structure_logits = self.ss8_head(x)
        sasa_logits = self.sasa_head(x)
        function_logits = self.function_head(x)
        function_logits = einops.rearrange(function_logits, '... (k v) -> ... k v', k=8)
        residue_logits = self.residue_head(x)
        return ESMOutput(sequence_logits=sequence_logits, structure_logits=structure_logits, secondary_structure_logits=secondary_structure_logits, sasa_logits=sasa_logits, function_logits=function_logits, residue_logits=residue_logits, embeddings=embed)

class ESM3(nn.Module, ESM3InferenceClient):

    def __init__(self, d_model: int, n_heads: int, v_heads: int, n_layers: int, structure_encoder_fn: Callable[[torch.device | str], StructureTokenEncoder], structure_decoder_fn: Callable[[torch.device | str], StructureTokenDecoder], function_decoder_fn: Callable[[torch.device | str], FunctionTokenDecoder], tokenizers: TokenizerCollectionProtocol):
        super().__init__()
        self.encoder = EncodeInputs(d_model)
        self.transformer = TransformerStack(d_model, n_heads, v_heads, n_layers, mask_and_zero_frameless=True)
        self.output_heads = OutputHeads(d_model)
        self.structure_encoder_fn = structure_encoder_fn
        self.structure_decoder_fn = structure_decoder_fn
        self.function_decoder_fn = function_decoder_fn
        self._structure_encoder = None
        self._structure_decoder = None
        self._function_decoder = None
        self.tokenizers = tokenizers

    @classmethod
    def from_pretrained(cls, model_name: str=ESM3_OPEN_SMALL, device: torch.device | None=None) -> ...:
        from esm.pretrained import load_local_model
        model_name = normalize_model_name(model_name)
        if not model_name:...
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = load_local_model(model_name, device=device)
        if device.type != 'cpu':
            model = model.to(torch.bfloat16)
        assert isinstance(model, ESM3)
        return model

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def raw_model(self):...

    def get_structure_encoder(self) -> StructureTokenEncoder:
        if self._structure_encoder is None:
            self._structure_encoder = self.structure_encoder_fn(self.device)
        return self._structure_encoder

    def get_structure_decoder(self) -> StructureTokenDecoder:
        if self._structure_decoder is None:
            self._structure_decoder = self.structure_decoder_fn(self.device)
        return self._structure_decoder

    def get_function_decoder(self) -> FunctionTokenDecoder:
        if self._function_decoder is None:
            self._function_decoder = self.function_decoder_fn(self.device)
        return self._function_decoder

    def forward(self, *, sequence_tokens: torch.Tensor | None=None, structure_tokens: torch.Tensor | None=None, ss8_tokens: torch.Tensor | None=None, sasa_tokens: torch.Tensor | None=None, function_tokens: torch.Tensor | None=None, residue_annotation_tokens: torch.Tensor | None=None, average_plddt: torch.Tensor | None=None, per_res_plddt: torch.Tensor | None=None, structure_coords: torch.Tensor | None=None, chain_id: torch.Tensor | None=None, sequence_id: torch.Tensor | None=None) -> ...:...
        try:
            (L, device) = next(((x.shape[1], x.device) for x in [sequence_tokens, structure_tokens, ss8_tokens, sasa_tokens, structure_coords, function_tokens, residue_annotation_tokens] if x is not None))
        except StopIteration:
            raise ValueError('At least one of the inputs must be non-None')
        t = self.tokenizers
        defaults = lambda x, tok: torch.full((1, L), tok, dtype=torch.long, device=device) if x is None else x
        sequence_tokens = defaults(sequence_tokens, t.sequence.mask_token_id)
        ss8_tokens = defaults(ss8_tokens, C.SS8_PAD_TOKEN)
        sasa_tokens = defaults(sasa_tokens, C.SASA_PAD_TOKEN)
        average_plddt = defaults(average_plddt, 1).float()
        per_res_plddt = defaults(per_res_plddt, 0).float()
        chain_id = defaults(chain_id, 0)
        if residue_annotation_tokens is None:...
        if function_tokens is None:...
        if structure_coords is None:...
        structure_coords = structure_coords[..., :3, :]
        (affine, affine_mask) = build_affine3d_from_coordinates(structure_coords)
        structure_tokens = defaults(structure_tokens, C.STRUCTURE_MASK_TOKEN)
        assert structure_tokens is not None
        structure_tokens = structure_tokens.masked_fill(structure_tokens == -1, C.STRUCTURE_MASK_TOKEN).masked_fill(sequence_tokens == C.SEQUENCE_BOS_TOKEN, C.STRUCTURE_BOS_TOKEN).masked_fill(sequence_tokens == C.SEQUENCE_PAD_TOKEN, C.STRUCTURE_PAD_TOKEN).masked_fill(sequence_tokens == C.SEQUENCE_EOS_TOKEN, C.STRUCTURE_EOS_TOKEN).masked_fill(sequence_tokens == C.SEQUENCE_CHAINBREAK_TOKEN, C.STRUCTURE_CHAINBREAK_TOKEN)
        x = self.encoder(sequence_tokens, structure_tokens, average_plddt, per_res_plddt, ss8_tokens, sasa_tokens, function_tokens, residue_annotation_tokens)
        (x, embedding, _) = self.transformer(x, sequence_id, affine, affine_mask, chain_id)
        return self.output_heads(x, embedding)

    def generate(self, input: ProteinType, config: GenerationConfig) -> ProteinType:...
        proteins = self.batch_generate([input], [config])
        assert len(proteins) == 1
        return proteins[0]

    def batch_generate(self, inputs: list[ProteinType], configs: list[GenerationConfig]) -> ...:
        assert len(inputs) == len(configs), ...
        if inputs == []:...
        t = type(inputs[0])
        for i in range(1, len(inputs)):...
        if isinstance(inputs[0], ESMProtein):
            return iterative_sampling_raw(self, inputs, configs)
        elif isinstance(inputs[0], ESMProteinTensor):
            return iterative_sampling_tokens(self, inputs, configs, self.tokenizers)
        else:...

    def encode(self, input: ESMProtein) -> ESMProteinTensor:
        input = attr.evolve(input)
        sequence_tokens = None
        structure_tokens = None
        secondary_structure_tokens = None
        sasa_tokens = None
        function_tokens = None
        residue_annotation_tokens = None
        coordinates = None
        if input.sequence is not None:
            sequence_tokens = encoding.tokenize_sequence(input.sequence, self.tokenizers.sequence, add_special_tokens=True)
        if input.secondary_structure is not None:...
        if input.sasa is not None:...
        sequence_length = -1
        if sequence_tokens is not None:
            sequence_length = len(sequence_tokens)
        elif secondary_structure_tokens is not None:...
        elif sasa_tokens is not None:...
        if input.coordinates is not None:
            (coordinates, _, structure_tokens) = encoding.tokenize_structure(input.coordinates, self.get_structure_encoder(), structure_tokenizer=self.tokenizers.structure, reference_sequence=input.sequence or '', add_special_tokens=True)
            if sequence_length == -1:
                sequence_length = len(structure_tokens)
        if sequence_length == -1:...
        if input.function_annotations is not None:...
        return ESMProteinTensor(sequence=sequence_tokens, structure=structure_tokens, secondary_structure=secondary_structure_tokens, sasa=sasa_tokens, function=function_tokens, residue_annotations=residue_annotation_tokens, coordinates=coordinates).to(next(self.parameters()).device)

    def decode(self, input: ESMProteinTensor) -> ESMProtein:
        return decode_protein_tensor(input=input, tokenizers=self.tokenizers, structure_token_decoder=self.get_structure_decoder(), function_token_decoder=self.get_function_decoder())

    def logits(self, input: ESMProteinTensor | _BatchedESMProteinTensor, config: LogitsConfig=LogitsConfig()) -> ...:
        if not isinstance(input, _BatchedESMProteinTensor):...
        device = torch.device(input.device)
        if input.coordinates is None:...
        else:
            per_res_plddt = input.coordinates.isfinite().all(dim=-1).any(dim=-1).float()
        with torch.no_grad(), torch.autocast(enabled=True, device_type=device.type, dtype=torch.bfloat16) if device.type == 'cuda' else ...:
            output = self.forward(sequence_tokens=input.sequence, structure_tokens=input.structure, ss8_tokens=input.secondary_structure, sasa_tokens=input.sasa, function_tokens=input.function, residue_annotation_tokens=input.residue_annotations, average_plddt=torch.tensor(1.0, device=input.device), per_res_plddt=per_res_plddt, structure_coords=input.coordinates, chain_id=None, sequence_id=None)
        output = ESMOutput(**{k: v.to(device).to(torch.float32) for (k, v) in vars(output).items()})
        return LogitsOutput(logits=ForwardTrackData(sequence=output.sequence_logits if config.sequence else None, structure=output.structure_logits if config.structure else None, secondary_structure=output.secondary_structure_logits if config.secondary_structure else ..., sasa=output.sasa_logits if config.sasa else None, function=output.function_logits if config.function else None), residue_annotation_logits=output.residue_logits if config.residue_annotations else ..., embeddings=output.embeddings if config.return_embeddings else None)

    def forward_and_sample(self, input: ESMProteinTensor, sampling_configuration: SamplingConfig) -> ...:...
```

`/esm/esm/tokenization/function_tokenizer.py`:

```python
import re
import string
from functools import cache, cached_property, partial
from typing import Collection
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from esm.tokenization.tokenizer_base import EsmTokenizerBase
from esm.utils.constants import esm3 as C
from esm.utils.function import interpro, lsh, tfidf
from esm.utils.misc import stack_variable_length_tensors
from esm.utils.types import FunctionAnnotation, PathLike

def _default_data_path(x: PathLike | None, d: PathLike) -> PathLike:...

def _default_local_data_path(x: PathLike | None, d: PathLike) -> PathLike:
    return x if x is not None else d

class InterProQuantizedTokenizer(EsmTokenizerBase):

    def __init__(self, depth: int=8, lsh_bits_per_token: int=8, lsh_path: PathLike | None=None, keyword_vocabulary_path: PathLike | None=None, keyword_idf_path: PathLike | None=None, interpro_entry_path: PathLike | None=None, interpro2keywords_path: PathLike | None=None):...
        self.depth = depth
        self.keyword_vocabulary_path = _default_local_data_path(keyword_vocabulary_path, C.KEYWORDS_VOCABULARY)
        self.keyword_idf_path = _default_local_data_path(keyword_idf_path, C.KEYWORDS_IDF)
        self._interpro2keywords_path = _default_local_data_path(interpro2keywords_path, C.INTERPRO2KEYWORDS)
        self.interpro_ = interpro.InterPro(entries_path=_default_local_data_path(interpro_entry_path, C.INTERPRO_ENTRY))
        self.lsh_path = lsh_path
        self.lsh_bits_per_token = lsh_bits_per_token
        self.lsh_vocab_size = 1 << lsh_bits_per_token
        self._lsh_token_vocab_offset = len(self.special_tokens) + 1

    @cached_property
    def _lsh(self) -> lsh.LSHTokenized:...

    @cached_property
    def interpro2keywords(self) -> dict[str, list[str]]:...

    @cached_property
    def interpro_labels(self) -> list[str]:...

    @cached_property
    def interpro_to_index(self) -> dict[str, int]:...

    @property
    def keyword_vocabulary(self) -> list[str]:...

    @property
    def keyword_to_index(self) -> dict[str, int]:...

    @cached_property
    def _tfidf(self) -> tfidf.TFIDFModel:...

    @cached_property
    def special_tokens(self) -> list[str]:...
        return ['<pad>', '<motif>', '<unk>']

    @cached_property
    def vocab(self) -> list[str]:...
        lsh_tokens = [f'<lsh:{i}>' for i in range(self.lsh_vocab_size)]
        return self.special_tokens + ['<none>'] + lsh_tokens

    @cached_property
    def vocab_to_index(self) -> dict[str, int]:
        return {token: token_id for (token_id, token) in enumerate(self.vocab)}

    def get_special_tokens_mask(self, encoded: torch.Tensor) -> torch.Tensor:...

    def tokenize(self, annotations: list[FunctionAnnotation], seqlen: int, p_keyword_dropout: float=0.0) -> list[str]:...

    def _function_text_hash(self, labels: Collection[str], keyword_mask: np.ndarray | None=None) -> np.ndarray | None:...

    def encode(self, tokens: list[str], add_special_tokens: bool=True) -> torch.Tensor:...

    def lookup_annotation_name(self, annotation: FunctionAnnotation) -> str | None:...

    def format_annotation(self, annotation: FunctionAnnotation) -> str:...

    def _token2ids(self, token: str) -> list[int]:...

    def batch_encode(self, token_batch: list[list[str]], add_special_tokens: bool=True) -> torch.Tensor:...

    def decode(self, encoded: torch.Tensor):...

    @property
    def mask_token(self) -> str:
        return '<pad>'

    @property
    def mask_token_id(self) -> int:
        return self.vocab_to_index[self.mask_token]

    @property
    def bos_token(self) -> str:
        return '<pad>'

    @property
    def bos_token_id(self) -> int:
        return self.vocab_to_index[self.bos_token]

    @property
    def eos_token(self) -> str:
        return '<pad>'

    @property
    def eos_token_id(self) -> int:
        return self.vocab_to_index[self.eos_token]

    @property
    def pad_token(self) -> str:
        return '<pad>'

    @property
    def pad_token_id(self) -> int:
        return self.vocab_to_index[self.pad_token]

    @property
    def chain_break_token(self) -> str:...

    @property
    def chain_break_token_id(self) -> int:...

    @property
    def all_token_ids(self):...

    @property
    def special_token_ids(self):...

def _texts_to_keywords(texts: list[str]) -> list[str]:...

def _keywords_from_text(text: str) -> list[str]:...

def _sanitize(text: str) -> str:...
_EXCLUDED_TERMS = {...}
```

`/esm/esm/tokenization/residue_tokenizer.py`:

```python
from functools import cached_property
from typing import Any
import pandas as pd
import torch
import torch.nn.functional as F
from cloudpathlib import AnyPath
from esm.tokenization.tokenizer_base import EsmTokenizerBase
from esm.utils.constants import esm3 as C
Sample = dict[str, Any]

class ResidueAnnotationsTokenizer(EsmTokenizerBase):

    def __init__(self, csv_path: str | None=None, max_annotations: int=16):
        if csv_path is None:
            csv_path = str(C.data_root('esm3') / C.RESID_CSV)
        self.csv_path = csv_path
        self.max_annotations = max_annotations

    @cached_property
    def _description2label(self) -> dict[str, str]:...

    @cached_property
    def _labels(self) -> list[str]:
        with AnyPath(self.csv_path).open() as f:
            df = pd.read_csv(f)
        labels = df.groupby('label_clean')['count'].sum().sort_values(ascending=False, kind='stable').index.tolist()
        assert isinstance(labels, list)
        return labels

    def _description2id(self, description: str) -> int | None:...

    @cached_property
    def _label2id(self) -> dict[str, int]:
        offset = len(self.special_tokens) + 1
        return {label: offset + i for (i, label) in enumerate(self._labels)}

    @cached_property
    def special_tokens(self) -> list[str]:...
        return ['<pad>', '<motif>', '<unk>']

    @cached_property
    def vocab(self):
        annotation_tokens = [f'<ra:{id}>' for (_, id) in self._label2id.items()]
        return self.special_tokens + ['<none>'] + annotation_tokens

    @cached_property
    def vocab_to_index(self) -> dict[str, int]:
        return {token: token_id for (token_id, token) in enumerate(self.vocab)}

    @cached_property
    def vocabulary(self) -> list[str]:...

    def get_special_tokens_mask(self, encoded: torch.Tensor) -> torch.Tensor:...

    def tokenize(self, sample: Sample | None, sequence: str, fail_on_mismatch: bool=False) -> list[str]:...

    def _token2ids(self, token: str) -> list[int]:...

    def encode(self, tokens: list[str], add_special_tokens: bool=True) -> torch.Tensor:...

    def decode(self, encoded: torch.Tensor) -> list[str]:...

    @property
    def mask_token(self) -> str:
        return '<pad>'

    @property
    def mask_token_id(self) -> int:
        return self.vocab_to_index[self.mask_token]

    @property
    def bos_token(self) -> str:
        return '<pad>'

    @property
    def bos_token_id(self) -> int:
        return self.vocab_to_index[self.bos_token]

    @property
    def eos_token(self) -> str:
        return '<pad>'

    @property
    def eos_token_id(self) -> int:
        return self.vocab_to_index[self.eos_token]

    @property
    def pad_token(self) -> str:
        return '<pad>'

    @property
    def pad_token_id(self) -> int:
        return self.vocab_to_index[self.pad_token]

    @property
    def chain_break_token(self) -> str:...

    @property
    def chain_break_token_id(self) -> int:...

    @property
    def all_token_ids(self):...

    @property
    def special_token_ids(self):...
```

`/esm/esm/sdk/forge.py`:

```python
import base64
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from functools import wraps
from typing import Literal, Sequence
from urllib.parse import urljoin
import requests
import torch
from attr import asdict
from tenacity import retry, retry_if_result, stop_after_attempt, wait_exponential
from esm.sdk.api import MSA, ESM3InferenceClient, ESMProtein, ESMProteinError, ESMProteinTensor, ForwardAndSampleOutput, ForwardTrackData, GenerationConfig, InverseFoldingConfig, LogitsConfig, LogitsOutput, ProteinType, SamplingConfig, SamplingTrackConfig
from esm.utils.misc import deserialize_tensors, maybe_list, maybe_tensor
from esm.utils.sampling import validate_sampling_config
from esm.utils.types import FunctionAnnotation
skip_retries_var = ContextVar('skip_retries', default=False)

def _list_to_function_annotations(l) -> list[FunctionAnnotation] | None:...

def retry_if_specific_error(exception):...

def log_retry_attempt(retry_state):...

def _validate_protein_tensor_input(input):...

class SequenceStructureForgeInferenceClient:

    def __init__(self, url: str='https://forge.evolutionaryscale.ai', model: str | None=None, token: str='', request_timeout: int | None=None):...

    def _fetch_msa(self, sequence: str) -> MSA:...

    def fold(self, sequence: str, msa: MSA | Literal['auto'] | None=None, potential_sequence_of_concern: bool=False, model_name: str | None=None) -> ESMProtein | ESMProteinError:...

    def inverse_fold(self, coordinates: torch.Tensor, config: InverseFoldingConfig, potential_sequence_of_concern: bool, model_name: str | None=None) -> ESMProtein | ESMProteinError:...

    def _post(self, endpoint, request, params={}, potential_sequence_of_concern: bool=False):...

class ESM3ForgeInferenceClient(ESM3InferenceClient):

    def __init__(self, model: str, url: str='https://forge.evolutionaryscale.ai', token: str='', request_timeout: int | None=None, min_retry_wait: int=1, max_retry_wait: int=10, max_retry_attempts: int=5):...

    @staticmethod
    def retry_decorator(func):...

        @wraps(func)
        def wrapper(instance, *args, **kwargs):...
        return wrapper

    @retry_decorator
    def generate(self, input: ProteinType, config: GenerationConfig) -> ProteinType:...

    def batch_generate(self, inputs: Sequence[ProteinType], configs: Sequence[GenerationConfig]) -> Sequence[ProteinType]:...

    def __generate_protein(self, input: ESMProtein, config: GenerationConfig) -> ESMProtein | ESMProteinError:...

    def __generate_protein_tensor(self, input: ESMProteinTensor, config: GenerationConfig) -> ESMProteinTensor | ESMProteinError:...

    @retry_decorator
    def forward_and_sample(self, input: ESMProteinTensor, sampling_configuration: SamplingConfig) -> ForwardAndSampleOutput | ESMProteinError:...

    @retry_decorator
    def encode(self, input: ESMProtein) -> ESMProteinTensor | ESMProteinError:...

    @retry_decorator
    def decode(self, input: ESMProteinTensor) -> ESMProtein | ESMProteinError:...

    @retry_decorator
    def logits(self, input: ESMProteinTensor, config: LogitsConfig=LogitsConfig(), return_bytes: bool=True) -> LogitsOutput | ESMProteinError:...

    def _post(self, endpoint, request, potential_sequence_of_concern, return_bytes: bool=False):...

    @property
    def raw_model(self):...
```

`/esm/esm/tokenization/__init__.py`:

```python
from dataclasses import dataclass
from typing import Protocol
from esm.utils.constants.models import ESM3_OPEN_SMALL, normalize_model_name
from .function_tokenizer import InterProQuantizedTokenizer
from .residue_tokenizer import ResidueAnnotationsTokenizer
from .sasa_tokenizer import SASADiscretizingTokenizer
from .sequence_tokenizer import EsmSequenceTokenizer
from .ss_tokenizer import SecondaryStructureTokenizer
from .structure_tokenizer import StructureTokenizer
from .tokenizer_base import EsmTokenizerBase

class TokenizerCollectionProtocol(Protocol):
    sequence: EsmSequenceTokenizer
    structure: StructureTokenizer
    secondary_structure: SecondaryStructureTokenizer
    sasa: SASADiscretizingTokenizer
    function: InterProQuantizedTokenizer
    residue_annotations: ResidueAnnotationsTokenizer

@dataclass
class TokenizerCollection:
    sequence: EsmSequenceTokenizer
    structure: StructureTokenizer
    secondary_structure: SecondaryStructureTokenizer
    sasa: SASADiscretizingTokenizer
    function: InterProQuantizedTokenizer
    residue_annotations: ResidueAnnotationsTokenizer

def get_esm3_model_tokenizers(model: str=ESM3_OPEN_SMALL) -> TokenizerCollection:
    if normalize_model_name(model) == ESM3_OPEN_SMALL:
        return TokenizerCollection(sequence=EsmSequenceTokenizer(), structure=StructureTokenizer(), secondary_structure=SecondaryStructureTokenizer(kind='ss8'), sasa=SASADiscretizingTokenizer(), function=InterProQuantizedTokenizer(), residue_annotations=ResidueAnnotationsTokenizer())
    else:...

def get_esmc_model_tokenizers() -> EsmSequenceTokenizer:...

def get_invalid_tokenizer_ids(tokenizer: EsmTokenizerBase) -> list[int]:...
```

`/esm/esm/tokenization/ss_tokenizer.py`:

```python
from functools import cached_property
from typing import Sequence
import torch
from esm.tokenization.tokenizer_base import EsmTokenizerBase
from esm.utils.constants import esm3 as C

class SecondaryStructureTokenizer(EsmTokenizerBase):

    def __init__(self, kind: str='ss8'):
        assert kind in ('ss8', 'ss3')
        self.kind = kind

    @property
    def special_tokens(self) -> list[str]:
        return ['<pad>', '<motif>', '<unk>']

    @cached_property
    def vocab(self):...
        match self.kind:
            case 'ss8':
                nonspecial_tokens = list(C.SSE_8CLASS_VOCAB)
            case 'ss3':...
            case _:...
        return [*self.special_tokens, *nonspecial_tokens]

    @cached_property
    def vocab_to_index(self) -> dict[str, int]:...
        return {word: i for (i, word) in enumerate(self.vocab)}

    def get_special_tokens_mask(self, tokens: torch.Tensor) -> torch.Tensor:...

    def encode(self, sequence: str | Sequence[str], add_special_tokens: bool=True) -> torch.Tensor:...

    def decode(self, encoded: torch.Tensor) -> str:...

    @property
    def mask_token(self) -> str:
        return '<pad>'

    @property
    def mask_token_id(self) -> int:
        return self.vocab_to_index[self.mask_token]

    @property
    def bos_token(self) -> str:
        return '<pad>'

    @property
    def bos_token_id(self) -> int:
        return self.vocab_to_index[self.bos_token]

    @property
    def eos_token(self) -> str:
        return '<pad>'

    @property
    def eos_token_id(self) -> int:
        return self.vocab_to_index[self.eos_token]

    @property
    def pad_token(self) -> str:
        return '<pad>'

    @property
    def pad_token_id(self) -> int:
        return self.vocab_to_index[self.pad_token]

    @property
    def chain_break_token(self) -> str:...

    @property
    def chain_break_token_id(self) -> int:...

    @property
    def all_token_ids(self):...

    @property
    def special_token_ids(self):...
```

`/esm/esm/utils/constants/physics.py`:

```python
BB_COORDINATES = [[0.5256, 1.3612, 0.0], [0.0, 0.0, 0.0], [-1.5251, 0.0, 0.0]]
```

`/esm/esm/utils/function/tfidf.py`:

```python
from collections import Counter
from functools import cached_property
import numpy as np
from cloudpathlib import AnyPath
from scipy import sparse
from esm.utils.types import PathLike

class TFIDFModel:

    def __init__(self, vocabulary_path: PathLike, idf_path: PathLike):...

    @cached_property
    def vocab_to_index(self) -> dict[str, int]:...

    def encode(self, terms: list[str]) -> sparse.csr_matrix:...

    def decode(self, vec: sparse.csr_matrix) -> list[str]:...
```

`/esm/esm/utils/encoding.py`:

```python
from typing import Sequence
import torch
import torch.nn.functional as F
from esm.models.vqvae import StructureTokenEncoder
from esm.tokenization.function_tokenizer import InterProQuantizedTokenizer as EsmFunctionTokenizer
from esm.tokenization.residue_tokenizer import ResidueAnnotationsTokenizer
from esm.tokenization.sasa_tokenizer import SASADiscretizingTokenizer
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
from esm.tokenization.ss_tokenizer import SecondaryStructureTokenizer
from esm.tokenization.structure_tokenizer import StructureTokenizer
from esm.utils.constants import esm3 as C
from esm.utils.function.encode_decode import encode_function_annotations
from esm.utils.structure.protein_chain import ProteinChain
from esm.utils.types import FunctionAnnotation

def get_default_sequence(sequence_length: int) -> str:...

def get_default_secondary_structure(sequence_length: int) -> str:...

def get_default_sasa(sequence_length: int) -> Sequence[float | str | None]:...

def tokenize_sequence(sequence: str, sequence_tokenizer: EsmSequenceTokenizer, add_special_tokens: bool=True) -> torch.Tensor:
    sequence = sequence.replace(C.MASK_STR_SHORT, sequence_tokenizer.mask_token)
    sequence_tokens = sequence_tokenizer.encode(sequence, add_special_tokens=add_special_tokens)
    sequence_tokens = torch.tensor(sequence_tokens, dtype=torch.int64)
    return sequence_tokens

def tokenize_structure(coordinates: torch.Tensor, structure_encoder: StructureTokenEncoder, structure_tokenizer: StructureTokenizer, reference_sequence: str='', add_special_tokens: bool=True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = next(structure_encoder.parameters()).device
    chain = ProteinChain.from_atom37(coordinates, sequence=reference_sequence if reference_sequence else None)
    if reference_sequence and len(reference_sequence) != coordinates.size(0):...
    left_pad = 0
    right_pad = 0
    if add_special_tokens:
        left_pad += 1
        right_pad += 1
    (coordinates, plddt, residue_index) = chain.to_structure_encoder_inputs()
    coordinates = coordinates.to(device)
    plddt = plddt.to(device)
    residue_index = residue_index.to(device)
    (_, structure_tokens) = structure_encoder.encode(coordinates, residue_index=residue_index)
    coordinates = torch.squeeze(coordinates, dim=0)
    plddt = torch.squeeze(plddt, dim=0)
    structure_tokens = torch.squeeze(structure_tokens, dim=0)
    if add_special_tokens:
        coordinates = F.pad(coordinates, (0, 0, 0, 0, left_pad, right_pad), value=torch.inf)
        plddt = F.pad(plddt, (left_pad, right_pad), value=0)
        structure_tokens = F.pad(structure_tokens, (left_pad, right_pad), value=structure_tokenizer.mask_token_id)
        structure_tokens[0] = structure_tokenizer.bos_token_id
        structure_tokens[-1] = structure_tokenizer.eos_token_id
    return (coordinates, plddt, structure_tokens)

def tokenize_secondary_structure(secondary_structure: str | Sequence[str], secondary_structure_tokenizer: SecondaryStructureTokenizer, add_special_tokens: bool=True) -> torch.Tensor:...

def tokenize_sasa(sasa: Sequence[float | str | None], sasa_tokenizer: SASADiscretizingTokenizer, add_special_tokens: bool=True):...

def tokenize_function_annotations(function_annotations: Sequence[FunctionAnnotation], reference_sequence: str, function_tokenizer: EsmFunctionTokenizer, residue_annotation_tokenizer: ResidueAnnotationsTokenizer, add_special_tokens: bool=True) -> tuple[torch.Tensor, torch.Tensor]:...

def get_default_sequence_tokens(sequence_length: int, sequence_tokenizer: EsmSequenceTokenizer) -> torch.Tensor:...

def get_default_structure_tokens(sequence_length: int, structure_tokenizer: StructureTokenizer) -> torch.Tensor:...

def get_default_secondary_structure_tokens(sequence_length: int, secondary_structure_tokenizer: SecondaryStructureTokenizer) -> torch.Tensor:...

def get_default_sasa_tokens(sequence_length: int, sasa_tokenizer: SASADiscretizingTokenizer) -> torch.Tensor:...

def get_default_function_tokens(sequence_length: int, function_tokenizer: EsmFunctionTokenizer) -> torch.Tensor:...

def get_default_residue_annotation_tokens(sequence_length: int, residue_annotation_tokenizer: ResidueAnnotationsTokenizer) -> torch.Tensor:...
```

`/esm/esm/utils/decoding.py`:

```python
import warnings
from typing import cast
import attr
import torch
from esm.models.function_decoder import FunctionTokenDecoder
from esm.models.vqvae import StructureTokenDecoder
from esm.sdk.api import ESMProtein, ESMProteinTensor
from esm.tokenization import TokenizerCollectionProtocol
from esm.tokenization.function_tokenizer import InterProQuantizedTokenizer
from esm.tokenization.residue_tokenizer import ResidueAnnotationsTokenizer
from esm.tokenization.sasa_tokenizer import SASADiscretizingTokenizer
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
from esm.tokenization.ss_tokenizer import SecondaryStructureTokenizer
from esm.tokenization.structure_tokenizer import StructureTokenizer
from esm.tokenization.tokenizer_base import EsmTokenizerBase
from esm.utils.constants import esm3 as C
from esm.utils.function.encode_decode import decode_function_tokens, decode_residue_annotation_tokens
from esm.utils.misc import maybe_list
from esm.utils.structure.protein_chain import ProteinChain
from esm.utils.types import FunctionAnnotation

def decode_protein_tensor(input: ESMProteinTensor, tokenizers: TokenizerCollectionProtocol, structure_token_decoder: StructureTokenDecoder, function_token_decoder: FunctionTokenDecoder | None=None) -> ESMProtein:
    input = attr.evolve(input)
    sequence = None
    secondary_structure = None
    sasa = None
    function_annotations = []
    coordinates = None
    for track in attr.fields(ESMProteinTensor):
        tokens: torch.Tensor | None = getattr(input, track.name)
        if track.name == 'coordinates' or track.name == 'potential_sequence_of_concern':
            continue
        if tokens is not None:
            tokens = tokens[1:-1]
            tokens = tokens.flatten()
            track_tokenizer = getattr(tokenizers, track.name)
            if torch.all(tokens == track_tokenizer.pad_token_id):
                setattr(input, track.name, None)
            if track.name == 'structure' and torch.any(tokens == track_tokenizer.mask_token_id):...
    if input.sequence is not None:
        sequence = decode_sequence(input.sequence, tokenizers.sequence)
    (plddt, ptm) = (None, None)
    if input.structure is not None:
        (coordinates, plddt, ptm) = decode_structure(structure_tokens=input.structure, structure_decoder=structure_token_decoder, structure_tokenizer=tokenizers.structure, sequence=sequence)
    elif input.coordinates is not None:...
    if input.secondary_structure is not None:...
    if input.sasa is not None:...
    if input.function is not None:...
    if input.residue_annotations is not None:...
    return ESMProtein(sequence=sequence, secondary_structure=secondary_structure, sasa=sasa, function_annotations=function_annotations if function_annotations else None, coordinates=coordinates, plddt=plddt, ptm=ptm, potential_sequence_of_concern=input.potential_sequence_of_concern)

def _bos_eos_warn(msg: str, tensor: torch.Tensor, tok: EsmTokenizerBase):
    if tensor[0] != tok.bos_token_id:...
    if tensor[-1] != tok.eos_token_id:...

def decode_sequence(sequence_tokens: torch.Tensor, sequence_tokenizer: EsmSequenceTokenizer, **kwargs) -> str:
    _bos_eos_warn('Sequence', sequence_tokens, sequence_tokenizer)
    sequence = sequence_tokenizer.decode(sequence_tokens, **kwargs)
    sequence = sequence.replace(' ', '')
    sequence = sequence.replace(sequence_tokenizer.mask_token, C.MASK_STR_SHORT)
    sequence = sequence.replace(sequence_tokenizer.cls_token, '')
    sequence = sequence.replace(sequence_tokenizer.eos_token, '')
    return sequence

def decode_structure(structure_tokens: torch.Tensor, structure_decoder: StructureTokenDecoder, structure_tokenizer: StructureTokenizer, sequence: str | None=None) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    is_singleton = len(structure_tokens.size()) == 1
    if is_singleton:
        structure_tokens = structure_tokens.unsqueeze(0)
    else:...
    _bos_eos_warn('Structure', structure_tokens[0], structure_tokenizer)
    decoder_output = structure_decoder.decode(structure_tokens)
    bb_coords: torch.Tensor = decoder_output['bb_pred'][0, 1:-1, ...]
    bb_coords = bb_coords.detach().cpu()
    if 'plddt' in decoder_output:
        plddt = decoder_output['plddt'][0, 1:-1]
        plddt = plddt.detach().cpu()
    else:...
    if 'ptm' in decoder_output:
        ptm = decoder_output['ptm']
    else:...
    chain = ProteinChain.from_backbone_atom_coordinates(bb_coords, sequence=sequence)
    chain = chain.infer_oxygen()
    return (torch.tensor(chain.atom37_positions), plddt, ptm)

def decode_secondary_structure(secondary_structure_tokens: torch.Tensor, ss_tokenizer: SecondaryStructureTokenizer) -> str:...

def decode_sasa(sasa_tokens: torch.Tensor, sasa_tokenizer: SASADiscretizingTokenizer) -> list[float]:...

def decode_function_annotations(function_annotation_tokens: torch.Tensor, function_token_decoder: FunctionTokenDecoder, function_tokenizer: InterProQuantizedTokenizer, **kwargs) -> list[FunctionAnnotation]:...

def decode_residue_annotations(residue_annotation_tokens: torch.Tensor, residue_annotation_decoder: ResidueAnnotationsTokenizer) -> list[FunctionAnnotation]:...
```

`/esm/esm/utils/structure/predicted_aligned_error.py`:

```python
import torch
import torch.nn.functional as F
from esm.utils.structure.affine3d import Affine3D

def masked_mean(mask: torch.Tensor, value: torch.Tensor, dim: int | None | tuple[int, ...]=None, eps=1e-10) -> torch.Tensor:...
    mask = mask.expand(*value.shape)
    return torch.sum(mask * value, dim=dim) / (eps + torch.sum(mask, dim=dim))

def _pae_bins(max_bin: float=31, num_bins: int=64, device: torch.device=torch.device('cpu')):
    bins = torch.linspace(0, max_bin, steps=num_bins - 1, device=device)
    step = max_bin / (num_bins - 2)
    bin_centers = bins + step / 2
    bin_centers = torch.cat([bin_centers, (bin_centers[-1] + step).unsqueeze(-1)], dim=0)
    return bin_centers

def _compute_pae_masks(mask: torch.Tensor):
    square_mask = (mask.unsqueeze(-1) * mask.unsqueeze(-2)).bool()
    return square_mask

def compute_predicted_aligned_error(logits: torch.Tensor, aa_mask: torch.Tensor, sequence_id: torch.Tensor | None=None, max_bin: float=31) -> torch.Tensor:
    bins = _pae_bins(max_bin, logits.shape[-1], logits.device)
    square_mask = _compute_pae_masks(aa_mask)
    min_v = torch.finfo(logits.dtype).min
    probs = logits.masked_fill(~square_mask.unsqueeze(-1), min_v).softmax(dim=-1)
    return (probs * bins).sum(dim=-1)

@torch.no_grad
def compute_tm(logits: torch.Tensor, aa_mask: torch.Tensor, max_bin: float=31.0):
    square_mask = _compute_pae_masks(aa_mask)
    seqlens = aa_mask.sum(-1, keepdim=True)
    bins = _pae_bins(max_bin, logits.shape[-1], logits.device)
    d0 = 1.24 * (seqlens.clamp_min(19) - 15) ** (1 / 3) - 1.8
    f_d = 1.0 / (1 + (bins / d0.unsqueeze(-1)) ** 2)
    min_v = torch.finfo(logits.dtype).min
    probs = logits.masked_fill(~square_mask.unsqueeze(-1), min_v).softmax(dim=-1)
    ptm = (probs * f_d.unsqueeze(-2)).sum(dim=-1)
    ptm = masked_mean(square_mask, ptm, dim=-1)
    return ptm.max(dim=-1).values

def tm_loss(logits: torch.Tensor, pred_affine: torch.Tensor, targ_affine: torch.Tensor, targ_mask: torch.Tensor, tm_mask: torch.Tensor | None=None, sequence_id: torch.Tensor | None=None, max_bin: float=31):...
```

`/esm/esm/layers/structure_proj.py`:

```python
import torch
import torch.nn as nn
from esm.utils.constants.physics import BB_COORDINATES
from esm.utils.structure.affine3d import Affine3D, RotationMatrix

class Dim6RotStructureHead(nn.Module):

    def __init__(self, input_dim: int, trans_scale_factor: float=10, norm_type: str='layernorm', activation_fn: str='esm_gelu', predict_torsion_angles: bool=True):
        super().__init__()
        self.ffn1 = nn.Linear(input_dim, input_dim)
        self.activation_fn = nn.GELU()
        self.norm = nn.LayerNorm(input_dim)
        self.proj = nn.Linear(input_dim, 9 + 7 * 2)
        self.trans_scale_factor = trans_scale_factor
        self.predict_torsion_angles = predict_torsion_angles
        self.bb_local_coords = torch.tensor(BB_COORDINATES).float()

    def forward(self, x, affine, affine_mask, **kwargs):
        if affine is None:
            rigids = Affine3D.identity(x.shape[:-1], dtype=x.dtype, device=x.device, requires_grad=self.training, rotation_type=RotationMatrix)
        else:...
        x = self.ffn1(x)
        x = self.activation_fn(x)
        x = self.norm(x)
        (trans, x, y, angles) = self.proj(x).split([3, 3, 3, 7 * 2], dim=-1)
        trans = trans * self.trans_scale_factor
        x = x / (x.norm(dim=-1, keepdim=True) + 1e-05)
        y = y / (y.norm(dim=-1, keepdim=True) + 1e-05)
        update = Affine3D.from_graham_schmidt(x + trans, trans, y + trans)
        rigids = rigids.compose(update.mask(affine_mask))
        affine = rigids.tensor
        all_bb_coords_local = self.bb_local_coords[None, None, :, :].expand(*x.shape[:-1], 3, 3).to(x.device)
        pred_xyz = rigids[..., None].apply(all_bb_coords_local)
        return (affine, pred_xyz)
```

`/esm/quickstart.py`:

```python
from huggingface_hub import login
from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient, ESMProtein, GenerationConfig
login()
model: ESM3InferenceClient = ESM3.from_pretrained('esm3-open').to('cuda')
prompt = '___________________________________________________DQATSLRILNNGHAFNVEFDDSQDKAVLKGGPLDGTYRLIQFHFHWGSLDGQGSEHTVDKKKYAAELHLVHWNTKYGDFGKAVQQPDGLAVLGIFLKVGSAKPGLQKVVDVLDSIKTKGKSADFTNFDPRGLLPESLDYWTYPGSLTTPP___________________________________________________________'
protein = ESMProtein(sequence=prompt)
protein = model.generate(protein, GenerationConfig(track='sequence', num_steps=8, temperature=0.7))
protein = model.generate(protein, GenerationConfig(track='structure', num_steps=8))
protein.to_pdb('./generation.pdb')
protein.sequence = None
protein = model.generate(protein, GenerationConfig(track='sequence', num_steps=8))
protein.coordinates = None
protein = model.generate(protein, GenerationConfig(track='structure', num_steps=8))
protein.to_pdb('./round_tripped.pdb')
```

`/esm/esm/utils/constants/esm3.py`:

```python
import os
from functools import cache
from pathlib import Path
from huggingface_hub import snapshot_download
SEQUENCE_BOS_TOKEN = 0
SEQUENCE_PAD_TOKEN = 1
SEQUENCE_EOS_TOKEN = 2
SEQUENCE_CHAINBREAK_TOKEN = 31
SEQUENCE_MASK_TOKEN = 32
VQVAE_CODEBOOK_SIZE = 4096
VQVAE_SPECIAL_TOKENS = {'MASK': VQVAE_CODEBOOK_SIZE, 'EOS': VQVAE_CODEBOOK_SIZE + 1, 'BOS': VQVAE_CODEBOOK_SIZE + 2, 'PAD': VQVAE_CODEBOOK_SIZE + 3, 'CHAINBREAK': VQVAE_CODEBOOK_SIZE + 4}
VQVAE_DIRECTION_LOSS_BINS = 16
VQVAE_PAE_BINS = 64
VQVAE_MAX_PAE_BIN = 31.0
VQVAE_PLDDT_BINS = 50
STRUCTURE_MASK_TOKEN = VQVAE_SPECIAL_TOKENS['MASK']
STRUCTURE_BOS_TOKEN = VQVAE_SPECIAL_TOKENS['BOS']
STRUCTURE_EOS_TOKEN = VQVAE_SPECIAL_TOKENS['EOS']
STRUCTURE_PAD_TOKEN = VQVAE_SPECIAL_TOKENS['PAD']
STRUCTURE_CHAINBREAK_TOKEN = VQVAE_SPECIAL_TOKENS['CHAINBREAK']
STRUCTURE_UNDEFINED_TOKEN = 955
SASA_PAD_TOKEN = 0
SS8_PAD_TOKEN = 0
INTERPRO_PAD_TOKEN = 0
RESIDUE_PAD_TOKEN = 0
CHAIN_BREAK_STR = '|'
SEQUENCE_BOS_STR = '<cls>'
SEQUENCE_EOS_STR = '<eos>'
MASK_STR_SHORT = '_'
SEQUENCE_MASK_STR = '<mask>'
SASA_MASK_STR = '<unk>'
SS8_MASK_STR = '<unk>'
SEQUENCE_VOCAB = [...]
SSE_8CLASS_VOCAB = 'GHITEBSC'
SSE_3CLASS_VOCAB = 'HEC'
SSE_8CLASS_TO_3CLASS_MAP = {'G': 'H', 'H': 'H', 'I': 'H', 'T': 'C', 'E': 'E', 'B': 'E', 'S': 'C', 'C': 'C'}
SASA_DISCRETIZATION_BOUNDARIES = [...]
MAX_RESIDUE_ANNOTATIONS = 16
TFIDF_VECTOR_SIZE = 58641

@staticmethod
@cache
def data_root(model: str):
    if 'INFRA_PROVIDER' in os.environ:...
    if model.startswith('esm3'):
        path = Path(snapshot_download(repo_id='EvolutionaryScale/esm3-sm-open-v1'))
    else:...
    return path
IN_REPO_DATA_FOLDER = Path(__file__).parents[2] / 'data'
INTERPRO_ENTRY = IN_REPO_DATA_FOLDER / 'entry_list_safety_29026.list'
INTERPRO_HIERARCHY = IN_REPO_DATA_FOLDER / 'ParentChildTreeFile.txt'
INTERPRO2GO = IN_REPO_DATA_FOLDER / 'ParentChildTreeFile.txt'
INTERPRO_2ID = 'data/tag_dict_4_safety_filtered.json'
LSH_TABLE_PATHS = {'8bit': 'data/hyperplanes_8bit_58641.npz'}
KEYWORDS_VOCABULARY = IN_REPO_DATA_FOLDER / 'keyword_vocabulary_safety_filtered_58641.txt'
KEYWORDS_IDF = IN_REPO_DATA_FOLDER / 'keyword_idf_safety_filtered_58641.npy'
RESID_CSV = 'data/uniref90_and_mgnify90_residue_annotations_gt_1k_proteins.csv'
INTERPRO2KEYWORDS = IN_REPO_DATA_FOLDER / 'interpro_29026_to_keywords_58641.csv'
```

`/esm/esm/sdk/__init__.py`:

```python
import os
from esm.sdk.forge import ESM3ForgeInferenceClient
from esm.utils.forge_context_manager import ForgeBatchExecutor

def client(model='esm3-sm-open-v1', url='https://forge.evolutionaryscale.ai', token=os.environ.get('ESM_API_KEY', ''), request_timeout=None):...

def batch_executor(max_attempts: int=10):...
```

`/esm/esm/utils/structure/protein_complex.py`:

```python
from __future__ import annotations
import io
import itertools
import re
import warnings
from dataclasses import asdict, dataclass, replace
from functools import cached_property
from pathlib import Path
from subprocess import check_output
from tempfile import TemporaryDirectory
from typing import Any, Iterable, Sequence
import biotite.structure as bs
import brotli
import msgpack
import msgpack_numpy
import numpy as np
import torch
from biotite.database import rcsb
from biotite.structure.io.pdb import PDBFile
from esm.utils import residue_constants
from esm.utils.constants import esm3 as esm3_c
from esm.utils.misc import slice_python_object_as_numpy
from esm.utils.structure.affine3d import Affine3D
from esm.utils.structure.aligner import Aligner
from esm.utils.structure.metrics import compute_gdt_ts, compute_lddt_ca
from esm.utils.structure.protein_chain import PathOrBuffer, ProteinChain
from esm.utils.structure.protein_structure import index_by_atom_name
msgpack_numpy.patch()
SINGLE_LETTER_CHAIN_IDS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'

def protein_chain_to_protein_complex(chain: ProteinChain) -> ProteinComplex:...

@dataclass
class ProteinComplexMetadata:
    entity_lookup: dict[int, int]
    chain_lookup: dict[int, str]
    chain_boundaries: list[tuple[int, int]]

@dataclass
class DockQSingleScore:
    native_chains: tuple[str, str]
    DockQ: float
    interface_rms: float
    ligand_rms: float
    fnat: float
    fnonnat: float
    clashes: float
    F1: float
    DockQ_F1: float

@dataclass
class DockQResult:
    total_dockq: float
    native_interfaces: int
    chain_mapping: dict[str, str]
    interfaces: dict[tuple[str, str], DockQSingleScore]
    aligned: ProteinComplex
    aligned_rmsd: float

class AtomIndexer:

    def __init__(self, structure: ProteinComplex, property: str, dim: int):
        self.structure = structure
        self.property = property
        self.dim = dim

    def __getitem__(self, atom_names: str | list[str]) -> np.ndarray:
        return index_by_atom_name(getattr(self.structure, self.property), atom_names, self.dim)

@dataclass
class ProteinComplex:
    id: str
    sequence: str
    entity_id: np.ndarray
    chain_id: np.ndarray
    sym_id: np.ndarray
    residue_index: np.ndarray
    insertion_code: np.ndarray
    atom37_positions: np.ndarray
    atom37_mask: np.ndarray
    confidence: np.ndarray
    metadata: ProteinComplexMetadata

    def __post_init__(self):
        l = len(self.sequence)
        assert self.atom37_positions.shape[0] == l, (self.atom37_positions.shape, l)
        assert self.atom37_mask.shape[0] == l, (self.atom37_mask.shape, l)
        assert self.residue_index.shape[0] == l, (self.residue_index.shape, l)
        assert self.insertion_code.shape[0] == l, (self.insertion_code.shape, l)
        assert self.confidence.shape[0] == l, (self.confidence.shape, l)
        assert self.entity_id.shape[0] == l, (self.entity_id.shape, l)
        assert self.chain_id.shape[0] == l, (self.chain_id.shape, l)
        assert self.sym_id.shape[0] == l, (self.sym_id.shape, l)

    def __getitem__(self, idx: int | list[int] | slice | np.ndarray):...
        if isinstance(idx, int):...
        if isinstance(idx, list):...
        if isinstance(idx, np.ndarray):...
        complex = self._unsafe_slice(idx)
        if len(complex) == 0:...
        chainbreak_runs = np.asarray([complex.sequence[i:i + 2] == '||' for i in range(len(complex.sequence) - 1)] + [complex.sequence[-1] == '|'])
        for i in range(len(chainbreak_runs)):
            if complex.sequence[i] == '|':...
            else:
                break
        complex = complex._unsafe_slice(~chainbreak_runs)
        return complex

    def _unsafe_slice(self, idx: int | list[int] | slice | np.ndarray):
        sequence = slice_python_object_as_numpy(self.sequence, idx)
        return replace(self, sequence=sequence, entity_id=self.entity_id[..., idx], chain_id=self.chain_id[..., idx], sym_id=self.sym_id[..., idx], residue_index=self.residue_index[..., idx], insertion_code=self.insertion_code[..., idx], atom37_positions=self.atom37_positions[..., idx, :, :], atom37_mask=self.atom37_mask[..., idx, :], confidence=self.confidence[..., idx])

    def __len__(self):
        return len(self.sequence)

    @cached_property
    def atoms(self) -> AtomIndexer:
        return AtomIndexer(self, property='atom37_positions', dim=-2)

    def chain_iter(self) -> Iterable[ProteinChain]:
        boundaries = [i for (i, s) in enumerate(self.sequence) if s == '|']
        boundaries = [-1, *boundaries, len(self)]
        for i in range(len(boundaries) - 1):
            c = self.__getitem__(slice(boundaries[i] + 1, boundaries[i + 1]))
            yield c.as_chain()

    def as_chain(self, force_conversion: bool=False) -> ProteinChain:...
        if not force_conversion:
            assert len(np.unique(self.chain_id)) == 1, f'{self.id}'
            assert len(np.unique(self.entity_id)) == 1, f'{self.id}'
            if self.chain_id[0] not in self.metadata.chain_lookup:...
            if self.entity_id[0] not in self.metadata.entity_lookup:
                warnings.warn('Entity ID not found in metadata, using None as default')
            chain_id = self.metadata.chain_lookup.get(self.chain_id[0], 'A')
            entity_id = self.metadata.entity_lookup.get(self.entity_id[0], None)
        else:...
        return ProteinChain(id=self.id, sequence=self.sequence, chain_id=chain_id, entity_id=entity_id, atom37_positions=self.atom37_positions, atom37_mask=self.atom37_mask, residue_index=self.residue_index, insertion_code=self.insertion_code, confidence=self.confidence)

    @classmethod
    def from_pdb(cls, path: PathOrBuffer, id: str | None=None) -> 'ProteinComplex':...

    @classmethod
    def from_rcsb(cls, pdb_id: str):...

    def to_pdb(self, path: PathOrBuffer, include_insertions: bool=True):
        atom_array = None
        for chain in self.chain_iter():
            carr = chain.atom_array if include_insertions else ...
            atom_array = carr if atom_array is None else atom_array + carr
        f = PDBFile()
        f.set_structure(atom_array)
        f.write(path)

    def to_pdb_string(self, include_insertions: bool=True) -> str:...

    def normalize_chain_ids_for_pdb(self):...

    def state_dict(self, backbone_only=False):...

    def to_blob(self, backbone_only=False) -> bytes:...

    @classmethod
    def from_state_dict(cls, dct):...

    @classmethod
    def from_blob(cls, input: Path | str | io.BytesIO | bytes):...

    @classmethod
    def from_chains(cls, chains: Sequence[ProteinChain]):
        if not chains:...

        def join_arrays(arrays: Sequence[np.ndarray], sep: np.ndarray):
            full_array = []
            for array in arrays:
                full_array.append(array)
                full_array.append(sep)
            full_array = full_array[:-1]
            return np.concatenate(full_array, 0)
        sep_tokens = {'residue_index': np.array([-1]), 'insertion_code': np.array(['']), 'atom37_positions': np.full([1, 37, 3], np.nan), 'atom37_mask': np.zeros([1, 37], dtype=bool), 'confidence': np.array([0])}
        array_args: dict[str, np.ndarray] = {name: join_arrays([getattr(chain, name) for chain in chains], sep) for (name, sep) in sep_tokens.items()}
        multimer_arrays = []
        chain2num_max = -1
        chain2num = {}
        ent2num_max = -1
        ent2num = {}
        total_index = 0
        chain_boundaries = []
        for (i, c) in enumerate(chains):
            num_res = c.residue_index.shape[0]
            if c.chain_id not in chain2num:
                chain2num[c.chain_id] = (chain2num_max := (chain2num_max + 1))
            chain_id_array = np.full([num_res], chain2num[c.chain_id], dtype=np.int64)
            if c.entity_id is None:
                entity_num = (ent2num_max := (ent2num_max + 1))
            else:...
            entity_id_array = np.full([num_res], entity_num, dtype=np.int64)
            sym_id_array = np.full([num_res], i, dtype=np.int64)
            multimer_arrays.append({'chain_id': chain_id_array, 'entity_id': entity_id_array, 'sym_id': sym_id_array})
            chain_boundaries.append((total_index, total_index + num_res))
            total_index += num_res + 1
        sep = np.array([-1])
        update = {name: join_arrays([dct[name] for dct in multimer_arrays], sep=sep) for name in ['chain_id', 'entity_id', 'sym_id']}
        array_args.update(update)
        metadata = ProteinComplexMetadata(chain_boundaries=chain_boundaries, chain_lookup={v: k for (k, v) in chain2num.items()}, entity_lookup={v: k for (k, v) in ent2num.items()})
        return cls(id=chains[0].id, sequence=esm3_c.CHAIN_BREAK_STR.join((chain.sequence for chain in chains)), metadata=metadata, **array_args)

    def infer_oxygen(self) -> ProteinComplex:...
        O_vector = torch.tensor([0.624, -1.0613, 0.0103], dtype=torch.float32)
        (N, CA, C) = torch.from_numpy(self.atoms[['N', 'CA', 'C']]).float().unbind(dim=1)
        N = torch.roll(N, -3)
        N[..., -1, :] = torch.nan
        frames = Affine3D.from_graham_schmidt(CA, C, N)
        O = frames.apply(O_vector)
        atom37_positions = self.atom37_positions.copy()
        atom37_mask = self.atom37_mask.copy()
        atom37_positions[:, residue_constants.atom_order['O']] = O.numpy()
        atom37_mask[:, residue_constants.atom_order['O']] = ~np.isnan(atom37_positions[:, residue_constants.atom_order['O']]).any(-1)
        new_chain = replace(self, atom37_positions=atom37_positions, atom37_mask=atom37_mask)
        return new_chain

    @classmethod
    def concat(cls, objs: list[ProteinComplex]) -> ProteinComplex:...

    def _sanity_check_complexes_are_comparable(self, other: ProteinComplex):...

    def lddt_ca(self, target: ProteinComplex, mobile_inds: list[int] | np.ndarray | None=None, target_inds: list[int] | np.ndarray | None=None, compute_chain_assignment: bool=True, **kwargs) -> ...:...

    def gdt_ts(self, target: ProteinComplex, mobile_inds: list[int] | np.ndarray | None=None, target_inds: list[int] | np.ndarray | None=None, compute_chain_assignment: bool=True, **kwargs) -> ...:...

    def dockq(self, native: ProteinComplex):...
```

`/esm/esm/models/esmc.py`:

```python
from __future__ import annotations
import contextlib
import attr
import torch
import torch.nn as nn
from attr import dataclass
try:
    from flash_attn.bert_padding import pad_input, unpad_input...
except ImportError:
    pad_input = None
    unpad_input = None
    is_flash_attn_available = False
from esm.layers.regression_head import RegressionHead
from esm.layers.transformer_stack import TransformerStack
from esm.sdk.api import ESMCInferenceClient, ESMProtein, ESMProteinTensor, ForwardTrackData, LogitsConfig, LogitsOutput
from esm.tokenization import EsmSequenceTokenizer
from esm.utils import encoding
from esm.utils.constants.models import ESMC_600M
from esm.utils.decoding import decode_sequence
from esm.utils.misc import stack_variable_length_tensors
from esm.utils.sampling import _BatchedESMProteinTensor

@dataclass
class ESMCOutput:
    sequence_logits: torch.Tensor
    embeddings: torch.Tensor | None
    hidden_states: torch.Tensor | None

class ESMC(nn.Module, ESMCInferenceClient):

    def __init__(self, d_model: int, n_heads: int, n_layers: int, tokenizer: EsmSequenceTokenizer, use_flash_attn: bool=True):...

    @classmethod
    def from_pretrained(cls, model_name: str=ESMC_600M, device: torch.device | None=None) -> ...:...

    @property
    def device(self):...

    @property
    def raw_model(self):...

    def _tokenize(self, sequence: list[str]) -> torch.Tensor:...

    def _detokenize(self, sequence: torch.Tensor) -> list[str]:...

    def forward(self, sequence_tokens: torch.Tensor | None=None, sequence_id: torch.Tensor | None=None) -> ...:...

    def encode(self, input: ESMProtein) -> ESMProteinTensor:...

    def decode(self, input: ESMProteinTensor) -> ESMProtein:...

    def logits(self, input: ESMProteinTensor | _BatchedESMProteinTensor, config: LogitsConfig=LogitsConfig()) -> ...:...
```

`/esm/esm/utils/structure/metrics.py`:

```python
import torch
from einops import rearrange
from esm.utils import residue_constants
from esm.utils.misc import unbinpack
from esm.utils.structure.protein_structure import compute_alignment_tensors, compute_gdt_ts_no_alignment

def compute_lddt(all_atom_pred_pos: torch.Tensor, all_atom_positions: torch.Tensor, all_atom_mask: torch.Tensor, cutoff: float=15.0, eps: float=1e-10, per_residue: bool=True, sequence_id: torch.Tensor | None=None) -> torch.Tensor:...

def compute_lddt_ca(all_atom_pred_pos: torch.Tensor, all_atom_positions: torch.Tensor, all_atom_mask: torch.Tensor, cutoff: float=15.0, eps: float=1e-10, per_residue: bool=True, sequence_id: torch.Tensor | None=None) -> torch.Tensor:...

def compute_gdt_ts(mobile: torch.Tensor, target: torch.Tensor, atom_exists_mask: torch.Tensor | None=None, sequence_id: torch.Tensor | None=None, reduction: str='per_sample'):...
```

`/esm/esm/layers/attention.py`:

```python
import functools
import einops
import torch
import torch.nn.functional as F
from torch import nn
from esm.layers.rotary import RotaryEmbedding, TritonRotaryEmbedding
try:
    from flash_attn import flash_attn_varlen_qkvpacked_func
except ImportError:
    flash_attn_varlen_func = None

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, n_heads: int, bias: bool=False, qk_layernorm: bool=True):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = self.d_model // self.n_heads
        self.layernorm_qkv = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model * 3, bias=bias))
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        if qk_layernorm:
            self.q_ln = nn.LayerNorm(d_model, bias=bias)
            self.k_ln = nn.LayerNorm(d_model, bias=bias)
        else:
            self.q_ln = nn.Identity()
            self.k_ln = nn.Identity()
        self.rotary = RotaryEmbedding(d_model // n_heads)

    def _apply_rotary(self, q: torch.Tensor, k: torch.Tensor):
        q = q.unflatten(-1, (self.n_heads, self.d_head))
        k = k.unflatten(-1, (self.n_heads, self.d_head))
        (q, k) = self.rotary(q, k)
        q = q.flatten(-2, -1)
        k = k.flatten(-2, -1)
        return (q, k)

    def forward(self, x, seq_id):
        qkv_BLD3 = self.layernorm_qkv(x)
        (query_BLD, key_BLD, value_BLD) = torch.chunk(qkv_BLD3, 3, dim=-1)
        (query_BLD, key_BLD) = (self.q_ln(query_BLD).to(query_BLD.dtype), self.k_ln(key_BLD).to(query_BLD.dtype))
        (query_BLD, key_BLD) = self._apply_rotary(query_BLD, key_BLD)
        reshaper = functools.partial(einops.rearrange, pattern='b s (h d) -> b h s d', h=self.n_heads)
        (query_BHLD, key_BHLD, value_BHLD) = map(reshaper, (query_BLD, key_BLD, value_BLD))
        if seq_id is not None:
            mask_BLL = seq_id.unsqueeze(-1) == seq_id.unsqueeze(-2)
            mask_BHLL = mask_BLL.unsqueeze(1)
            context_BHLD = F.scaled_dot_product_attention(query_BHLD, key_BHLD, value_BHLD, mask_BHLL)
        else:
            context_BHLD = F.scaled_dot_product_attention(query_BHLD, key_BHLD, value_BHLD)
        context_BLD = einops.rearrange(context_BHLD, 'b h s d -> b s (h d)')
        return self.out_proj(context_BLD)

class FlashMultiHeadAttention(MultiHeadAttention):

    def __init__(self, d_model: int, n_heads: int, bias: bool=False, qk_layernorm: bool=True):...

    def forward(self, x, seq_id):...
```

`/esm/esm/layers/codebook.py`:

```python
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

class EMACodebook(nn.Module):

    def __init__(self, n_codes, embedding_dim, no_random_restart=True, restart_thres=1.0, ema_decay=0.99):
        super().__init__()
        self.register_buffer('embeddings', torch.randn(n_codes, embedding_dim))
        self.register_buffer('N', torch.zeros(n_codes))
        self.register_buffer('z_avg', self.embeddings.data.clone())
        self.n_codes = n_codes
        self.embedding_dim = embedding_dim
        self._need_init = True
        self.no_random_restart = no_random_restart
        self.restart_thres = restart_thres
        self.freeze_codebook = False
        self.ema_decay = ema_decay

    def reset_parameters(self):...

    def _tile(self, x):...

    def _init_embeddings(self, z):...

    def forward(self, z):
        if self._need_init and self.training and (not self.freeze_codebook):...
        flat_inputs = z.view(-1, self.embedding_dim)
        distances = (flat_inputs ** 2).sum(dim=1, keepdim=True) - 2 * flat_inputs @ self.embeddings.t() + (self.embeddings.t() ** 2).sum(dim=0, keepdim=True)
        encoding_indices = torch.argmin(distances, dim=1)
        encoding_indices = encoding_indices.view(*z.shape[:2])
        embeddings = F.embedding(encoding_indices, self.embeddings)
        commitment_loss = 0.25 * F.mse_loss(z, embeddings.detach())
        if self.training and (not self.freeze_codebook):...
        embeddings_st = (embeddings - z).detach() + z
        return (embeddings_st, encoding_indices, commitment_loss)

    def dictionary_lookup(self, encodings):...

    def soft_codebook_lookup(self, weights: torch.Tensor) -> torch.Tensor:...
```

`/esm/esm/utils/structure/protein_chain.py`:

```python
from __future__ import annotations
import io
from dataclasses import asdict, dataclass, replace
from functools import cached_property
from pathlib import Path
from typing import Sequence, TypeVar, Union
import biotite.structure as bs
import brotli
import msgpack
import msgpack_numpy
import numpy as np
import torch
from Bio.Data import PDBData
from biotite.application.dssp import DsspApp
from biotite.database import rcsb
from biotite.structure.io.pdb import PDBFile
from cloudpathlib import CloudPath
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist, squareform
from torch import Tensor
from esm.utils import residue_constants as RC
from esm.utils.constants import esm3 as C
from esm.utils.misc import slice_python_object_as_numpy
from esm.utils.structure.affine3d import Affine3D
from esm.utils.structure.aligner import Aligner
from esm.utils.structure.metrics import compute_lddt_ca
from esm.utils.structure.normalize_coordinates import apply_frame_to_coords, get_protein_normalization_frame, normalize_coordinates
msgpack_numpy.patch()
CHAIN_ID_CONST = 'A'
ArrayOrTensor = TypeVar('ArrayOrTensor', np.ndarray, Tensor)
PathLike = Union[str, Path, CloudPath]
PathOrBuffer = Union[PathLike, io.StringIO]

def index_by_atom_name(atom37: ArrayOrTensor, atom_names: str | list[str], dim: int=-2) -> ...:
    squeeze = False
    if isinstance(atom_names, str):...
    indices = [RC.atom_order[atom_name] for atom_name in atom_names]
    dim = dim % atom37.ndim
    index = tuple((slice(None) if dim != i else indices for i in range(atom37.ndim)))
    result = atom37[index]
    if squeeze:...
    return result

def infer_CB(C, N, Ca, L: float=1.522, A: float=1.927, D: float=-2.143):...

class AtomIndexer:

    def __init__(self, structure: ProteinChain, property: str, dim: int):
        self.structure = structure
        self.property = property
        self.dim = dim

    def __getitem__(self, atom_names: str | list[str]) -> np.ndarray:
        return index_by_atom_name(getattr(self.structure, self.property), atom_names, self.dim)

@dataclass
class ProteinChain:
    id: str
    sequence: str
    chain_id: str
    entity_id: int | None
    residue_index: np.ndarray
    insertion_code: np.ndarray
    atom37_positions: np.ndarray
    atom37_mask: np.ndarray
    confidence: np.ndarray

    def __post_init__(self):
        self.atom37_mask = self.atom37_mask.astype(bool)
        assert self.atom37_positions.shape[0] == len(self.sequence), (...,)
        assert self.atom37_mask.shape[0] == len(self.sequence), (...,)
        assert self.residue_index.shape[0] == len(self.sequence), (...,)
        assert self.insertion_code.shape[0] == len(self.sequence), (...,)
        assert self.confidence.shape[0] == len(self.sequence), (...,)

    @cached_property
    def atoms(self) -> AtomIndexer:
        return AtomIndexer(self, property='atom37_positions', dim=-2)

    @cached_property
    def atom_mask(self) -> AtomIndexer:...

    @cached_property
    def atom_array(self) -> bs.AtomArray:
        atoms = []
        for (res_name, res_idx, ins_code, positions, mask, conf) in zip(self.sequence, self.residue_index, self.insertion_code, self.atom37_positions, self.atom37_mask.astype(bool), self.confidence):
            for (i, pos) in zip(np.where(mask)[0], positions[mask]):
                atom = bs.Atom(coord=pos, chain_id='A' if self.chain_id is None else self.chain_id, res_id=res_idx, ins_code=ins_code, res_name=RC.restype_1to3.get(res_name, 'UNK'), hetero=False, atom_name=RC.atom_types[i], element=RC.atom_types[i][0], b_factor=conf)
                atoms.append(atom)
        return bs.array(atoms)

    @cached_property
    def residue_index_no_insertions(self) -> np.ndarray:...

    @cached_property
    def atom_array_no_insertions(self) -> bs.AtomArray:...

    def __getitem__(self, idx: int | list[int] | slice | np.ndarray):...

    def __len__(self):...

    def cbeta_contacts(self, distance_threshold: float=8.0) -> np.ndarray:...

    def to_structure_encoder_inputs(self, should_normalize_coordinates: bool=True) -> ...:
        coords = torch.tensor(self.atom37_positions, dtype=torch.float32)
        plddt = torch.tensor(self.confidence, dtype=torch.float32)
        residue_index = torch.tensor(self.residue_index, dtype=torch.long)
        if should_normalize_coordinates:
            coords = normalize_coordinates(coords)
        return (coords.unsqueeze(0), plddt.unsqueeze(0), residue_index.unsqueeze(0))

    def to_pdb(self, path: PathOrBuffer, include_insertions: bool=True):...

    def to_pdb_string(self, include_insertions: bool=True) -> str:...

    def state_dict(self, backbone_only=False):...

    def to_blob(self, backbone_only=False) -> bytes:...

    @classmethod
    def from_state_dict(cls, dct):...

    @classmethod
    def from_blob(cls, input: Path | str | io.BytesIO | bytes):...

    def dssp(self):...

    def sasa(self):...

    def globularity(self) -> float:...

    @staticmethod
    def _mvee(P: np.ndarray, tol, max_iter=10000):...

    def align(self, target: ProteinChain, mobile_inds: list[int] | np.ndarray | None=None, target_inds: list[int] | np.ndarray | None=None, only_use_backbone: bool=False):...

    def rmsd(self, target: ProteinChain, also_check_reflection: bool=False, mobile_inds: list[int] | np.ndarray | None=None, target_inds: list[int] | np.ndarray | None=None, only_compute_backbone_rmsd: bool=False):...

    def lddt_ca(self, native: ProteinChain, mobile_inds: list[int] | np.ndarray | None=None, target_inds: list[int] | np.ndarray | None=None, **kwargs) -> ...:...

    @classmethod
    def from_atom37(cls, atom37_positions: np.ndarray | torch.Tensor, *, id: str | None=None, sequence: str | None=None, chain_id: str | None=None, entity_id: int | None=None, residue_index: np.ndarray | torch.Tensor | None=None, insertion_code: np.ndarray | None=None, confidence: np.ndarray | torch.Tensor | None=None):
        if isinstance(atom37_positions, torch.Tensor):
            atom37_positions = atom37_positions.cpu().numpy()
            if atom37_positions.ndim == 4:...
        assert isinstance(atom37_positions, np.ndarray)
        seqlen = atom37_positions.shape[0]
        atom_mask = np.isfinite(atom37_positions).all(-1)
        if id is None:
            id = ''
        if sequence is None:
            sequence = 'A' * seqlen
        if chain_id is None:
            chain_id = 'A'
        if residue_index is None:
            residue_index = np.arange(1, seqlen + 1)
        else:...
        assert isinstance(residue_index, np.ndarray)
        if insertion_code is None:
            insertion_code = np.array(['' for _ in range(seqlen)])
        if confidence is None:
            confidence = np.ones(seqlen, dtype=np.float32)
        else:...
        assert isinstance(confidence, np.ndarray)
        return cls(id=id, sequence=sequence, chain_id=chain_id, entity_id=entity_id, atom37_positions=atom37_positions, atom37_mask=atom_mask, residue_index=residue_index, insertion_code=insertion_code, confidence=confidence)

    @classmethod
    def from_backbone_atom_coordinates(cls, backbone_atom_coordinates: np.ndarray | torch.Tensor, **kwargs):...
        if isinstance(backbone_atom_coordinates, torch.Tensor):
            backbone_atom_coordinates = backbone_atom_coordinates.cpu().numpy()
            if backbone_atom_coordinates.ndim == 4:...
        assert isinstance(backbone_atom_coordinates, np.ndarray)
        assert backbone_atom_coordinates.ndim == 3
        assert backbone_atom_coordinates.shape[-2] == 3
        assert backbone_atom_coordinates.shape[-1] == 3
        atom37_positions = np.full((backbone_atom_coordinates.shape[0], 37, 3), np.inf, dtype=backbone_atom_coordinates.dtype)
        atom37_positions[:, :3, :] = backbone_atom_coordinates
        return cls.from_atom37(atom37_positions=atom37_positions, **kwargs)

    @classmethod
    def from_pdb(cls, path: PathOrBuffer, chain_id: str='detect', id: str | None=None, is_predicted: bool=False) -> ...:...

    @classmethod
    def from_rcsb(cls, pdb_id: str, chain_id: str='detect'):...

    @classmethod
    def from_atomarray(cls, atom_array: bs.AtomArray, id: str | None=None) -> ...:...

    def get_normalization_frame(self) -> Affine3D:...

    def apply_frame(self, frame: Affine3D) -> ProteinChain:...

    def normalize_coordinates(self) -> ProteinChain:...

    def infer_oxygen(self) -> ProteinChain:...
        O_vector = torch.tensor([0.624, -1.0613, 0.0103], dtype=torch.float32)
        (N, CA, C) = torch.from_numpy(self.atoms[['N', 'CA', 'C']]).float().unbind(dim=1)
        N = torch.roll(N, -3)
        N[..., -1, :] = torch.nan
        frames = Affine3D.from_graham_schmidt(CA, C, N)
        O = frames.apply(O_vector)
        atom37_positions = self.atom37_positions.copy()
        atom37_mask = self.atom37_mask.copy()
        atom37_positions[:, RC.atom_order['O']] = O.numpy()
        atom37_mask[:, RC.atom_order['O']] = ~np.isnan(atom37_positions[:, RC.atom_order['O']]).any(-1)
        new_chain = replace(self, atom37_positions=atom37_positions, atom37_mask=atom37_mask)
        return new_chain

    @cached_property
    def inferred_cbeta(self) -> np.ndarray:...

    def infer_cbeta(self, infer_cbeta_for_glycine: bool=False) -> ProteinChain:...

    @cached_property
    def pdist_CA(self) -> np.ndarray:...

    @cached_property
    def pdist_CB(self) -> np.ndarray:...

    @classmethod
    def as_complex(cls, chains: Sequence[ProteinChain]):...

    @classmethod
    def concat(cls, chains: Sequence[ProteinChain]):...

    def select_residue_indices(self, indices: list[int | str], ignore_x_mismatch: bool=False) -> ...:...
```

`/esm/esm/layers/blocks.py`:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from esm.layers.attention import FlashMultiHeadAttention, MultiHeadAttention
from esm.layers.geom_attention import GeometricReasoningOriginalImpl
from esm.utils.structure.affine3d import Affine3D

def swiglu_correction_fn(expansion_ratio: float, d_model: int) -> int:
    return int((expansion_ratio * d_model + 255) // 256 * 256)

class SwiGLU(nn.Module):

    def __init__(self):
        super(SwiGLU, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        (x1, x2) = x.chunk(2, dim=-1)
        return F.silu(x1) * x2

def swiglu_ln_ffn(d_model: int, expansion_ratio: float, bias: bool):
    return nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, swiglu_correction_fn(expansion_ratio, d_model) * 2, bias=bias), SwiGLU(), nn.Linear(swiglu_correction_fn(expansion_ratio, d_model), d_model, bias=bias))

def gelu_ln_ffn(d_model: int, expansion_ratio: float, bias: bool):
    hidden_dim = int(expansion_ratio * d_model)
    return nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, hidden_dim, bias=bias), nn.GELU(), nn.Linear(hidden_dim, d_model, bias=bias))

class UnifiedTransformerBlock(nn.Module):

    def __init__(self, d_model: int, n_heads: int, use_geom_attn: bool=False, use_plain_attn: bool=True, use_flash_attn: bool=False, v_heads: int | None=None, bias: bool=False, expansion_ratio: float=4.0, residue_scaling_factor: float=1, mask_and_zero_frameless: bool=False, qk_layernorm: bool=True, ffn_type: str='swiglu'):
        super().__init__()
        self.use_plain_attn = use_plain_attn
        if self.use_plain_attn:
            if use_flash_attn:...
            else:
                self.attn = MultiHeadAttention(d_model, n_heads, bias, qk_layernorm=qk_layernorm)
        self.use_geom_attn = use_geom_attn
        if self.use_geom_attn:
            if v_heads is None:...
            self.geom_attn = GeometricReasoningOriginalImpl(c_s=d_model, v_heads=v_heads, bias=bias, mask_and_zero_frameless=mask_and_zero_frameless)
        if ffn_type == 'swiglu':
            self.ffn = swiglu_ln_ffn(d_model, expansion_ratio, bias)
        elif ffn_type == 'gelu':
            self.ffn = gelu_ln_ffn(d_model, expansion_ratio, bias)
        else:...
        self.scaling_factor = residue_scaling_factor

    def forward(self, x: torch.Tensor, sequence_id: torch.Tensor, frames: Affine3D, frames_mask: torch.Tensor, chain_id: torch.Tensor) -> torch.Tensor:...
        if self.use_plain_attn:
            r1 = self.attn(x, sequence_id)
            x = x + r1 / self.scaling_factor
        if self.use_geom_attn:
            r2 = self.geom_attn(x, frames, frames_mask, sequence_id, chain_id)
            x = x + r2 / self.scaling_factor
        r3 = self.ffn(x) / self.scaling_factor
        x = x + r3
        return x
```

`/esm/esm/utils/sampling.py`:

```python
import warnings
from typing import Literal
import attr
import torch
import torch.nn.functional as F
from esm.sdk.api import ESMProteinTensor, SamplingConfig, SamplingTrackConfig
from esm.tokenization import TokenizerCollectionProtocol, get_invalid_tokenizer_ids
from esm.tokenization.function_tokenizer import InterProQuantizedTokenizer
from esm.utils.constants.esm3 import MAX_RESIDUE_ANNOTATIONS, SASA_DISCRETIZATION_BOUNDARIES

def _non_batched_dims(k: str, v: torch.Tensor):
    match k:
        case 'sequence':
            return 1
        case 'structure':
            if v.is_floating_point():...
            else:
                return 1
        case 'secondary_structure':
            return 1
        case 'sasa':
            return 1
        case 'function':
            return 2
        case 'residue_annotations':
            return 2
        case 'coordinates':
            return 3
        case _:...

class _BatchedESMProteinTensor(ESMProteinTensor):

    @staticmethod
    def from_protein_tensor(protein: ESMProteinTensor):

        def _maybe_unsqueeze(x: torch.Tensor | None):
            return x.unsqueeze(0) if x is not None else None
        return _BatchedESMProteinTensor(sequence=_maybe_unsqueeze(protein.sequence), structure=_maybe_unsqueeze(protein.structure), secondary_structure=_maybe_unsqueeze(protein.secondary_structure), sasa=_maybe_unsqueeze(protein.sasa), function=_maybe_unsqueeze(protein.function), residue_annotations=_maybe_unsqueeze(protein.residue_annotations), coordinates=_maybe_unsqueeze(protein.coordinates))

    def __len__(self) -> int:

        def get_len(k, v) -> int:
            assert len(v.shape) == _non_batched_dims(k, v) + 1
            return v.size(1)
        l = self._detect_attribute(get_len, 'length')
        return l if l is not None else 0

    @property
    def batch_size(self) -> int:...

    def slice(self, i: int, sequence_len: int | None=None) -> ESMProteinTensor:

        def _maybe_slice(x: torch.Tensor | None):
            if x is None:...
            row = x[i]
            if sequence_len is not None:
                row = row[:sequence_len]
            return row
        return ESMProteinTensor(sequence=_maybe_slice(self.sequence), structure=_maybe_slice(self.structure), secondary_structure=_maybe_slice(self.secondary_structure), sasa=_maybe_slice(self.sasa), function=_maybe_slice(self.function), residue_annotations=_maybe_slice(self.residue_annotations), coordinates=_maybe_slice(self.coordinates))

    def set_slice(self, i: int, slice: ESMProteinTensor):...

def get_default_sampling_config(tokenizers: TokenizerCollectionProtocol) -> SamplingConfig:...

def validate_sampling_config(sampling_config: SamplingConfig, on_invalid: Literal['raise', 'warn']='warn'):...

def sample_logits(logits: torch.Tensor, temperature: float | torch.Tensor, valid_ids: list[int]=[], top_p: float | torch.Tensor=1.0, mask_logits_of_invalid_ids: bool=True):...
    if len(valid_ids) == 0:...
    if top_p < 1.0:...
    temperature = _tensorize_like(temperature, logits)
    batch_dims = logits.size()[:-1]
    logits = logits.reshape(-1, logits.shape[-1])
    if mask_logits_of_invalid_ids:
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask[..., valid_ids] = False
        logits[mask] = -torch.inf
    if torch.all(temperature == 0):...
    assert not torch.any(temperature == 0), 'Partial temperature 0 not supported.'
    probs = F.softmax(logits / temperature[..., None], dim=-1)
    ids = torch.multinomial(probs, 1).squeeze(1)
    ids = ids.reshape(*batch_dims)
    return ids

def sample_function_logits(logits: torch.Tensor, tokenizer: InterProQuantizedTokenizer, top_p: float | torch.Tensor=1.0, temperature: float | torch.Tensor=1.0, p_none_threshold: float=0.05) -> tuple[torch.Tensor, torch.Tensor]:...

def sample_residue_annotation_logits(logits: torch.Tensor, annotation_threshold: float=0.5) -> tuple[torch.Tensor, torch.Tensor]:...

def sample_sasa_logits(logits: torch.Tensor, tokens: torch.Tensor, sampling_track_config: SamplingTrackConfig, mask_idx: int, valid_ids: list[int], mask_logits_of_invalid_ids: bool=True) -> torch.Tensor:...

def top_p_logits(logits: torch.Tensor, top_p: float | torch.Tensor) -> torch.Tensor:...

def _tensorize_like(value: int | float | torch.Tensor, logits: torch.Tensor):
    if isinstance(value, (float, int)):
        value = torch.full_like(logits[..., 0], value, dtype=logits.dtype)
    return value.to(logits.device).expand_as(logits[..., 0]).reshape(-1)

def get_sampling_mask(tokens: torch.Tensor, sampling_track_config: SamplingTrackConfig, mask_idx: int):
    sampling_mask = torch.ones_like(tokens, dtype=torch.bool)
    sampling_mask[:, 0] = False
    sampling_mask[:, -1] = False
    special_minus_mask = list(set(sampling_track_config.invalid_ids) - {mask_idx})
    if len(special_minus_mask) > 0:...
    if sampling_track_config.only_sample_masked_tokens:
        masked_tokens = tokens == mask_idx
        sampling_mask = sampling_mask & masked_tokens
    return sampling_mask
```

`/esm/esm/layers/transformer_stack.py`:

```python
import math
import torch
import torch.nn as nn
from esm.layers.blocks import UnifiedTransformerBlock
from esm.utils.structure.affine3d import Affine3D

class TransformerStack(nn.Module):

    def __init__(self, d_model: int, n_heads: int, v_heads: int | None, n_layers: int, n_layers_geom: int=1, scale_residue: bool=True, mask_and_zero_frameless: bool=False, bias: bool=False, qk_layernorm: bool=True, ffn_type: str='swiglu', expansion_ratio: float=8 / 3, use_flash_attn: bool=False):
        super().__init__()
        self.blocks = nn.ModuleList([UnifiedTransformerBlock(d_model, n_heads, v_heads=v_heads, use_geom_attn=i < n_layers_geom, use_flash_attn=use_flash_attn, residue_scaling_factor=math.sqrt(n_layers / 36) if scale_residue else 1.0, expansion_ratio=expansion_ratio, mask_and_zero_frameless=mask_and_zero_frameless, bias=bias, qk_layernorm=qk_layernorm, ffn_type=ffn_type) for i in range(n_layers)])
        self.norm = nn.LayerNorm(d_model, bias=False)

    def forward(self, x: torch.Tensor, sequence_id: torch.Tensor | None=None, affine: Affine3D | None=None, affine_mask: torch.Tensor | None=None, chain_id: torch.Tensor | None=None) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:...
        (*batch_dims, _) = x.shape
        if chain_id is None:...
        hiddens = []
        for block in self.blocks:
            x = block(x, sequence_id, affine, affine_mask, chain_id)
            hiddens.append(x)
        return (self.norm(x), x, hiddens)
```

`/esm/esm/utils/function/lsh.py`:

```python
import numpy as np
from cloudpathlib import AnyPath
from esm.utils.types import PathLike

class LSHTable:

    def __init__(self, n_bits: int, dim: int, hyperplanes: np.ndarray | None=None):...

    def __call__(self, array, tokenize: bool=True):...

class LSHTokenized:

    def __init__(self, n_bits: int, dim: int, num_tables: int=1, filepath: PathLike | None=None, allow_create_hyperplanes: bool=False):...

    def write_hyperplanes(self, filepath: PathLike):...

    def __call__(self, array):...

class LSHBitstream:

    def __init__(self, n_bits: int, dim: int, filepath: PathLike | None=None, allow_create_hyperplanes: bool=False):...

    def write_hyperplanes(self, filepath: PathLike):...

    def __call__(self, array):...
```

`/esm/esm/utils/constants/api.py`:

```python
MAX_TOPK_SEQUENCE = 32
MAX_TOPK_STRUCTURE = MAX_TOPK_SEQUENCE
MAX_TOPK_SECONDARY_STRUCTURE = MAX_TOPK_SEQUENCE
MAX_TOPK_SASA = MAX_TOPK_SEQUENCE
MAX_TOPK_FUNCTION = MAX_TOPK_SEQUENCE
```

`/esm/esm/utils/residue_constants.py`:

```python
atom_types = [...]
atom_order = {atom_type: i for (i, atom_type) in enumerate(atom_types)}
atom_type_num = len(atom_types)
restype_1to3 = {'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS', 'Q': 'GLN', 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO', 'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'}
amino_acid_volumes = {'A': 88.6, 'R': 173.4, 'N': 114.1, 'D': 111.1, 'C': 108.5, 'Q': 143.8, 'E': 138.4, 'G': 60.1, 'H': 153.2, 'I': 166.7, 'L': 166.7, 'K': 168.6, 'M': 162.9, 'F': 189.9, 'P': 112.7, 'S': 89.0, 'T': 116.1, 'W': 227.8, 'Y': 193.6, 'V': 140.0, 'X': 88.6}
```

`/esm/esm/utils/structure/normalize_coordinates.py`:

```python
from typing import TypeVar
import numpy as np
import torch
from torch import Tensor
from esm.utils import residue_constants as RC
from esm.utils.structure.affine3d import Affine3D
ArrayOrTensor = TypeVar('ArrayOrTensor', np.ndarray, Tensor)

def atom3_to_backbone_frames(bb_positions: torch.Tensor) -> Affine3D:
    (N, CA, C) = bb_positions.unbind(dim=-2)
    return Affine3D.from_graham_schmidt(C, CA, N)

def index_by_atom_name(atom37: ArrayOrTensor, atom_names: str | list[str], dim: int=-2) -> ArrayOrTensor:
    squeeze = False
    if isinstance(atom_names, str):...
    indices = [RC.atom_order[atom_name] for atom_name in atom_names]
    dim = dim % atom37.ndim
    index = tuple((slice(None) if dim != i else indices for i in range(atom37.ndim)))
    result = atom37[index]
    if squeeze:...
    return result

def get_protein_normalization_frame(coords: Tensor) -> Affine3D:...
    bb_coords = index_by_atom_name(coords, ['N', 'CA', 'C'], dim=-2)
    coord_mask = torch.all(torch.all(torch.isfinite(bb_coords), dim=-1), dim=-1)
    average_position_per_n_ca_c = bb_coords.masked_fill(~coord_mask[..., None, None], 0).sum(-3) / (coord_mask.sum(-1)[..., None, None] + 1e-08)
    frame = atom3_to_backbone_frames(average_position_per_n_ca_c.float())
    return frame

def apply_frame_to_coords(coords: Tensor, frame: Affine3D) -> Tensor:...
    coords_trans_rot = frame[..., None, None].invert().apply(coords)
    valid_frame = frame.trans.norm(dim=-1) > 0
    is_inf = torch.isinf(coords)
    coords = coords_trans_rot.where(valid_frame[..., None, None, None], coords)
    coords.masked_fill_(is_inf, torch.inf)
    return coords

def normalize_coordinates(coords: Tensor) -> Tensor:
    return apply_frame_to_coords(coords, get_protein_normalization_frame(coords))
```

`/esm/esm/utils/generation.py`:

```python
import os
from typing import Any, Callable, Sequence
from warnings import warn
import attr
import torch
from tqdm import tqdm
from esm.sdk.api import ESM3InferenceClient, ESMProtein, ESMProteinError, ESMProteinTensor, ForwardAndSampleOutput, ForwardTrackData, GenerationConfig, LogitsConfig, LogitsOutput, SamplingConfig, SamplingTrackConfig
from esm.tokenization import EsmTokenizerBase, TokenizerCollectionProtocol
from esm.tokenization.function_tokenizer import InterProQuantizedTokenizer
from esm.utils.constants import esm3 as C
from esm.utils.misc import stack_variable_length_tensors
from esm.utils.noise_schedules import NOISE_SCHEDULE_REGISTRY
from esm.utils.sampling import _BatchedESMProteinTensor, get_sampling_mask, sample_function_logits, sample_logits, sample_residue_annotation_logits, sample_sasa_logits

def _trim_sequence_tensor_dataclass(o: Any, sequence_len: int):...
    assert attr.has(o.__class__)
    sliced = {}
    for (k, v) in attr.asdict(o, recurse=False).items():
        if v is None:
            sliced[k] = None
        elif isinstance(v, torch.Tensor):
            sliced[k] = v[:, :sequence_len]
        elif isinstance(v, tuple) and all((isinstance(t, torch.Tensor) for t in v)):...
        elif attr.has(v.__class__):
            sliced[k] = _trim_sequence_tensor_dataclass(v, sequence_len)
        else:...
    return attr.evolve(o, **sliced)

def _slice_tensor_dataclass(o: Any, i: int, keep_dim: bool=False) -> Any:...
    assert attr.has(o.__class__)
    sliced = {}
    for (k, v) in attr.asdict(o, recurse=False).items():
        if v is None:
            sliced[k] = None
        elif isinstance(v, torch.Tensor):
            row = v.select(0, i)
            if keep_dim:
                row = row.unsqueeze(0)
            sliced[k] = row
        elif attr.has(v.__class__):
            sliced[k] = _slice_tensor_dataclass(v, i, keep_dim)
        else:...
    return attr.evolve(o, **sliced)

def iterative_sampling_raw(client: ESM3InferenceClient, proteins: list[ESMProtein], configs: list[GenerationConfig]) -> list[ESMProtein | ESMProteinError]:
    input_tokens = [client.encode(protein) for protein in proteins]
    output_tokens_list = client.batch_generate(input_tokens, configs)
    raw_proteins: list[ESMProtein | ESMProteinError] = []
    for output_tokens in output_tokens_list:
        if isinstance(output_tokens, ESMProteinTensor):
            raw_proteins.append(client.decode(output_tokens))
        else:...
    for (input_protein, raw_protein, config) in zip(proteins, raw_proteins, configs):
        if isinstance(raw_protein, ESMProteinError):...
        if config.track not in ['function', 'residue_annotations']:
            raw_protein.function_annotations = input_protein.function_annotations
    return raw_proteins

def _make_masked_inputs(track: str, sequence_length: int, tokenizers: TokenizerCollectionProtocol):
    get_tokenizer: Callable[[str], EsmTokenizerBase] = lambda s: getattr(tokenizers, s)
    has_tokenizer: Callable[[str], bool] = lambda s: hasattr(tokenizers, s)
    if track == 'coordinates':
        dims = (sequence_length, 3, 3)
    elif track == 'confidence':...
    elif track == 'attention_mask':...
    elif track == 'function':
        dims = (sequence_length, tokenizers.function.depth)
    elif track == 'residue_annotations':
        dims = (sequence_length, C.MAX_RESIDUE_ANNOTATIONS)
    else:
        dims = (sequence_length,)
    if track == 'coordinates':
        masked_tokens = torch.full(dims, torch.inf, dtype=torch.float)
    elif track == 'confidence':...
    elif track == 'attention_mask':...
    elif has_tokenizer(track):
        masked_tokens = torch.full(dims, get_tokenizer(track).mask_token_id, dtype=torch.long)
        masked_tokens[0] = get_tokenizer(track).bos_token_id
        masked_tokens[-1] = get_tokenizer(track).eos_token_id
    else:...
    return masked_tokens

def _stack_protein_tensors(input_tokens: list[ESMProteinTensor], sequence_lengths: list[int], tokenizers: TokenizerCollectionProtocol, device: str | torch.device) -> _BatchedESMProteinTensor:
    o = _BatchedESMProteinTensor()

    def _maybe_mock_input(fn, t, l):
        if t is not None:
            return t
        t = _make_masked_inputs(fn, l, tokenizers)
        if t is not None:
            t = t.to(device)
        return t

    def _stack_field(fn: str):
        tensors = [getattr(tokens, fn) for tokens in input_tokens]
        tensors = [_maybe_mock_input(fn, t, l) for (t, l) in zip(tensors, sequence_lengths)]
        if all([t is None for t in tensors]):...
        if fn == 'coordinates':
            mask_token_id = torch.inf
        else:
            mask_token_id = getattr(tokenizers, fn).pad_token_id
        setattr(o, fn, stack_variable_length_tensors(sequences=tensors, constant_value=mask_token_id))
    for f in attr.fields(ESMProteinTensor):
        if f.name == 'potential_sequence_of_concern':
            continue
        _stack_field(f.name)
    return o

def _get_masked_positions(track: str, tokens: torch.Tensor, mask_token_id: int) -> torch.Tensor:
    if track == 'function':...
    else:
        mask = tokens == mask_token_id
    mask[..., 0] = False
    mask[..., -1] = False
    return mask

def _get_iterative_sampling_mask_for_prompt_and_step(cur_sampled: _BatchedESMProteinTensor, sequence_lengths: torch.Tensor, total_to_sample: torch.Tensor, step: int, entropy: ForwardTrackData, config: GenerationConfig, tokenizers: TokenizerCollectionProtocol) -> torch.Tensor:...
    track_to_sample = config.track
    tokens = getattr(cur_sampled, track_to_sample)
    device = tokens.device
    shape = tokens.shape
    (B, L) = (shape[0], shape[1])
    assert B == 1
    sampling_mask = torch.ones((B, L), dtype=torch.bool, device=device)
    sampling_mask[:, 0] = False
    sampling_mask &= (torch.arange(L).repeat(B, 1) < (sequence_lengths - 1).unsqueeze(-1)).to(device)
    is_mask = _get_masked_positions(track_to_sample, tokens, getattr(tokenizers, track_to_sample).mask_token_id)
    if not is_mask.any().item():...
    sampling_mask = sampling_mask & is_mask
    decoding_schedule = NOISE_SCHEDULE_REGISTRY[config.schedule]
    still_masked = torch.sum(sampling_mask).int()
    perc_masked_after_this_step = decoding_schedule(torch.tensor((step + 1) / config.num_steps))
    num_tokens_masked_after_this_step = (perc_masked_after_this_step * total_to_sample + 0.1).int()
    num_to_sample = still_masked - num_tokens_masked_after_this_step
    if config.strategy == 'entropy':...
    elif config.strategy == 'random':
        (_, masked_indices) = sampling_mask.nonzero(as_tuple=True)
        rnd_indices = masked_indices[torch.randperm(len(masked_indices))][:num_to_sample]
        rnd_mask = torch.zeros_like(sampling_mask)
        rnd_mask[:, rnd_indices] = True
        where_to_sample = sampling_mask & rnd_mask
    if track_to_sample == 'function':...
    return where_to_sample

def _get_non_special_tokens(protein: ESMProteinTensor, tokenizers: TokenizerCollectionProtocol) -> int:
    if protein.sequence is None:
        return len(protein) - 2
    mask = torch.ones_like(protein.sequence)
    for special_token in tokenizers.sequence.special_token_ids:
        if special_token == tokenizers.sequence.mask_token_id:
            continue
        mask[protein.sequence == special_token] = 0
    return int(torch.sum(mask).item())

def _get_annealed_temperature(step: int, num_steps: int, initial_temperature: float):
    step_ratio = step / max(1, num_steps - 1)
    return max(initial_temperature - step_ratio, 0.001) ** 2

def iterative_sampling_tokens(client: ESM3InferenceClient, input_tokens: list[ESMProteinTensor], configs: list[GenerationConfig], tokenizers: TokenizerCollectionProtocol) -> Sequence[ESMProteinTensor | ESMProteinError]:
    devices = set([t.device for t in input_tokens])
    if len(devices) > 1:...
    sampled_tokens = [attr.evolve(tokens) for tokens in input_tokens]
    for (tokens, config) in zip(sampled_tokens, configs):
        if config.condition_on_coordinates_only and tokens.coordinates is not None:
            tokens.structure = None
    sequence_lengths = [len(tokens) for tokens in sampled_tokens]
    total_to_sample = []
    for (protein, config) in zip(sampled_tokens, configs):
        track = config.track
        if getattr(protein, track) is None:
            num_sampling_steps = _get_non_special_tokens(protein, tokenizers)
        else:
            masked = _get_masked_positions(track, getattr(protein, track), getattr(tokenizers, track).mask_token_id)
            num_sampling_steps = torch.sum(masked).item()
        total_to_sample.append(num_sampling_steps)
        if num_sampling_steps > 0 and num_sampling_steps < config.num_steps:...
    max_num_steps = max([config.num_steps for config in configs])
    batched_tokens = _stack_protein_tensors(sampled_tokens, sequence_lengths, tokenizers, devices.pop())
    errors: dict[int, ESMProteinError] = {}
    disable_tqdm = bool(os.environ.get('DISABLE_ITERATIVE_SAMPLING_TQDM', False))
    for t in tqdm(range(max_num_steps), disable=disable_tqdm):
        forward_out = _batch_forward(client, batched_tokens)
        for (i, config) in enumerate(configs):
            if i in errors:...
            if config.track in ['coordinates', 'residue_annotations']:...
            if t >= config.num_steps:...
            per_prompt_cur_sampled = _BatchedESMProteinTensor.from_protein_tensor(batched_tokens.slice(i))
            per_prompt_forward_out: LogitsOutput = _slice_tensor_dataclass(forward_out, i, keep_dim=True)
            per_prompt_forward_out = _trim_sequence_tensor_dataclass(per_prompt_forward_out, len(per_prompt_cur_sampled))
            if config.temperature_annealing:
                temperature = _get_annealed_temperature(t, config.num_steps, config.temperature)
            else:...
            track_sample_config = SamplingTrackConfig()
            track_sample_config.invalid_ids = config.invalid_ids
            track_sample_config.temperature = temperature
            track_sample_config.top_p = config.top_p
            sampling_config = SamplingConfig(**{config.track: track_sample_config})
            per_prompt_forward_and_sample_output = _sample_per_prompt(per_prompt_cur_sampled, per_prompt_forward_out, sampling_config, tokenizers, decode_sasa_tokens=False)
            per_prompt_new_sampled = per_prompt_forward_and_sample_output.protein_tensor
            assert per_prompt_forward_and_sample_output.entropy is not None
            try:
                where_to_sample = _get_iterative_sampling_mask_for_prompt_and_step(per_prompt_cur_sampled, torch.tensor(sequence_lengths[i]), torch.tensor(total_to_sample[i]), t, per_prompt_forward_and_sample_output.entropy, config, tokenizers)
            except ValueError as e:
                errors[i] = ESMProteinError(error_code=500, error_msg=str(e))
                continue
            where_to_sample.to(input_tokens[0].device)
            old_track_samples = getattr(per_prompt_cur_sampled, config.track)
            new_track_samples = getattr(per_prompt_new_sampled, config.track)
            new_track_samples = torch.where(where_to_sample, new_track_samples, old_track_samples)
            getattr(batched_tokens, config.track)[i, ...] = new_track_samples[0]
    output_tokens = [batched_tokens.slice(i, sequence_len=sequence_lengths[i]) if i not in errors else ... for i in range(len(input_tokens))]
    for (inputs, outputs, config) in zip(input_tokens, output_tokens, configs):
        if isinstance(outputs, ESMProteinError):...
        setattr(outputs, 'coordinates', getattr(inputs, 'coordinates'))
        for f in attr.fields(SamplingConfig):
            if 'embedding' in f.name or f.name == 'return_hidden_states':
                continue
            if f.name != config.track:
                setattr(outputs, f.name, getattr(inputs, f.name))
    return output_tokens

def _batch_forward(client: ESM3InferenceClient, protein: _BatchedESMProteinTensor):
    return client.logits(protein, LogitsConfig(sequence=True, structure=True, secondary_structure=True, sasa=True, function=True, residue_annotations=True, return_embeddings=True))

def _sample_per_prompt(protein: _BatchedESMProteinTensor, logits_output: LogitsOutput, sampling_config: SamplingConfig, tokenizers: TokenizerCollectionProtocol, decode_sasa_tokens: bool=True, mask_logits_of_invalid_ids: bool=True) -> ForwardAndSampleOutput:
    assert logits_output.logits is not None

    def maybe_clone(x: torch.Tensor | None) -> torch.Tensor | None:
        return x.clone() if x is not None else None
    tokens_dir = {}
    track_sampling_metadata_dir: dict[str, dict | None] = {}
    integer_sampling_tracks = ['sequence', 'structure', 'secondary_structure']
    if not decode_sasa_tokens:
        integer_sampling_tracks.append('sasa')
    for track in integer_sampling_tracks:
        config = getattr(sampling_config, track)
        if config is None:
            tokens_dir[track] = maybe_clone(getattr(protein, track))
            continue
        tokenizer = getattr(tokenizers, track)
        valid_ids = set(tokenizer.all_token_ids) - set(tokenizer.special_token_ids) - set(config.invalid_ids)
        sampling_metadata = _sample_track(logits=getattr(logits_output.logits, track), tokens=getattr(protein, track), sampling_track_config=config, mask_idx=getattr(tokenizers, track).mask_token_id, valid_ids=list(valid_ids), mask_logits_of_invalid_ids=mask_logits_of_invalid_ids)
        tokens_dir[track] = sampling_metadata.pop('sampled_tokens')
        track_sampling_metadata_dir[track] = sampling_metadata
    if decode_sasa_tokens:...
    config = getattr(sampling_config, 'function')
    function_logits = getattr(logits_output.logits, 'function')
    if config is None or function_logits is None:
        tokens_dir['function'] = maybe_clone(getattr(protein, 'function'))
        tokens_dir['residue_annotations'] = maybe_clone(getattr(protein, 'residue_annotations'))
    else:...
    forward_and_sample_output_dir = {}
    forward_and_sample_output_dir['protein_tensor'] = ESMProteinTensor(**tokens_dir)
    for property in [...]:
        is_all_none = True
        forward_track_data_dir = {}
        for track in track_sampling_metadata_dir.keys():
            values = track_sampling_metadata_dir[track]
            if values is not None and values.get(property, None) is not None:
                forward_track_data_dir[track] = values.get(property, None)
                is_all_none = False
        if not is_all_none:
            forward_and_sample_output_dir[property] = ForwardTrackData(**forward_track_data_dir)
        else:
            forward_and_sample_output_dir[property] = None
    per_res_embed = logits_output.embeddings if sampling_config.return_per_residue_embeddings else None
    mean_embedding = logits_output.embeddings.mean(dim=1) if sampling_config.return_mean_embedding else None
    return ForwardAndSampleOutput(per_residue_embedding=per_res_embed, mean_embedding=mean_embedding, **forward_and_sample_output_dir)

def _sample_track(logits: torch.Tensor, tokens: torch.Tensor, sampling_track_config: SamplingTrackConfig, mask_idx: int, valid_ids: list[int], mask_logits_of_invalid_ids: bool=True) -> dict[str, torch.Tensor]:...
    temperature = sampling_track_config.temperature
    sampled_tokens = sample_logits(logits, temperature=temperature, valid_ids=valid_ids, top_p=sampling_track_config.top_p, mask_logits_of_invalid_ids=mask_logits_of_invalid_ids)
    log_probs = logits.log_softmax(-1)
    sampling_mask = get_sampling_mask(tokens, sampling_track_config, mask_idx)
    sampled_tokens = torch.where(sampling_mask, sampled_tokens, tokens)
    return _compute_track_metadata(sampled_tokens, log_probs, sampling_mask, top_k=sampling_track_config.topk_logprobs)

def _sample_function_track(function_tokenizer: InterProQuantizedTokenizer, tokens: torch.Tensor, logits: torch.Tensor, sampling_track_config: SamplingTrackConfig) -> dict[str, torch.Tensor]:...

def _compute_track_metadata(sampled_tokens: torch.Tensor, log_probs: torch.Tensor, sampling_mask: torch.Tensor, top_k: int) -> dict:...
    probs = torch.exp(log_probs)
    entropy = torch.distributions.Categorical(logits=log_probs).entropy()
    sampled_logprob = torch.zeros_like(sampled_tokens, dtype=log_probs.dtype)
    if sampled_tokens.dim() > sampling_mask.dim():...
    sampled_tokens_valid = sampled_tokens[sampling_mask]
    sampled_log_probs_valid = log_probs[sampling_mask, sampled_tokens_valid]
    sampled_logprob[sampling_mask] = sampled_log_probs_valid
    sampled_prob = torch.exp(sampled_logprob)
    top_prob = torch.max(probs, dim=-1).values
    (topk_logprobs, topk_tokens) = torch.topk(log_probs, top_k, dim=-1)
    topk_logprobs = None if top_k == 0 else topk_logprobs
    topk_tokens = None if top_k == 0 else topk_tokens
    return {'entropy': entropy, 'sampled_tokens': sampled_tokens, 'prob': sampled_prob, 'logprob': sampled_logprob, 'top_prob': top_prob, 'topk_logprob': topk_logprobs, 'topk_tokens': topk_tokens}
```

`/esm/esm/utils/function/interpro.py`:

```python
import itertools
import re
from dataclasses import dataclass
from enum import IntEnum, auto
from functools import cached_property
import networkx as nx
import pandas as pd
from cloudpathlib import AnyPath
from esm.utils.constants import esm3 as C
from esm.utils.types import PathLike

def parse_go_terms(text: str) -> list[str]:...

def _parse_interpro2go(path: PathLike) -> dict[str, list[str]]:...

class InterProEntryType(IntEnum):
    ACTIVE_SITE = 0
    BINDING_SITE = auto()
    CONSERVED_SITE = auto()
    DOMAIN = auto()
    FAMILY = auto()
    HOMOLOGOUS_SUPERFAMILY = auto()
    PTM = auto()
    REPEAT = auto()
    UNKNOWN = auto()

@dataclass
class InterProEntry:
    id: str
    type: InterProEntryType
    name: str
    description: str | None = None

class InterPro:

    def __init__(self, entries_path: PathLike | None=None, hierarchy_path: PathLike | None=None, interpro2go_path: PathLike | None=None):...

        def default(x, d):
            return x if x is not None else d
        self.entries_path = default(entries_path, C.INTERPRO_ENTRY)
        self.hierarchy_graph_path = default(hierarchy_path, C.INTERPRO_HIERARCHY)
        self.interpro2go_path = default(interpro2go_path, C.INTERPRO2GO)

    @cached_property
    def interpro2go(self) -> dict[str, list[str]]:...

    @cached_property
    def entries_frame(self) -> pd.DataFrame:...

    @cached_property
    def entries(self) -> dict[str, InterProEntry]:...

    def lookup_name(self, interpro_id: str) -> str | None:...

    def lookup_entry_type(self, interpro_id: str) -> InterProEntryType:...

    @cached_property
    def graph(self) -> nx.DiGraph:...
```

`/esm/esm/utils/types.py`:

```python
from __future__ import annotations
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Union
from cloudpathlib import CloudPath
PathLike = Union[str, Path, CloudPath]
PathOrBuffer = Union[PathLike, io.StringIO]

@dataclass
class FunctionAnnotation:
    label: str
    start: int
    end: int

    def to_tuple(self) -> tuple[str, int, int]:...

    def __len__(self) -> int:...
```

`/esm/esm/tokenization/sasa_tokenizer.py`:

```python
from functools import cached_property
import torch
from esm.tokenization.tokenizer_base import EsmTokenizerBase
from esm.utils.constants import esm3 as C

class SASADiscretizingTokenizer(EsmTokenizerBase):

    def __init__(self, boundaries: list[float]=C.SASA_DISCRETIZATION_BOUNDARIES):
        self._boundaries = sorted(boundaries)

    @cached_property
    def special_tokens(self) -> list[str]:
        return ['<pad>', '<motif>', '<unk>']

    @cached_property
    def vocab(self) -> list[str]:...
        boundary_strs = ['0'] + [str(b) for b in self._boundaries] + ['inf']
        range_tokens = [f'<{low}-{high}>' for (low, high) in zip(boundary_strs[:-1], boundary_strs[1:])]
        return self.special_tokens + range_tokens

    @cached_property
    def midpoints_tensor(self) -> torch.Tensor:...

    def midpoints(self) -> list[float]:...

    @cached_property
    def vocab_to_index(self) -> dict[str, int]:...
        return {word: i for (i, word) in enumerate(self.vocab)}

    def get_special_tokens_mask(self, tokens: torch.Tensor) -> torch.Tensor:...

    def encode(self, values: list[float | str], add_special_tokens: bool=True) -> torch.Tensor:...

    def decode_float(self, encoded: torch.Tensor) -> list[float]:...

    def decode(self, encoded: torch.Tensor) -> str:...

    def decode_list(self, encoded: torch.Tensor) -> list[str]:...

    @property
    def mask_token(self) -> str:
        return '<pad>'

    @property
    def mask_token_id(self) -> int:
        return self.vocab_to_index[self.mask_token]

    @property
    def bos_token(self) -> str:
        return '<pad>'

    @property
    def bos_token_id(self) -> int:
        return self.vocab_to_index[self.bos_token]

    @property
    def eos_token(self) -> str:
        return '<pad>'

    @property
    def eos_token_id(self) -> int:
        return self.vocab_to_index[self.eos_token]

    @property
    def pad_token(self) -> str:
        return '<pad>'

    @property
    def pad_token_id(self) -> int:
        return self.vocab_to_index[self.pad_token]

    @property
    def chain_break_token(self) -> str:...

    @property
    def chain_break_token_id(self) -> int:...

    @property
    def all_token_ids(self):...

    @property
    def special_token_ids(self):...
```

