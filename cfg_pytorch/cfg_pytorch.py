from __future__ import annotations
from collections import namedtuple
from torch import nn, einsum, Tensor

from einops import rearrange, repeat, pack, unpack

from beartype.door import is_bearable
from beartype.typing import Callable, Tuple, List, Literal, Dict, Any

from inspect import signature
from cfg_pytorch.typing import typecheck, beartype_isinstance

from cfg_pytorch.t5 import T5Adapter
from cfg_pytorch.open_clip import OpenClipAdapter
from cfg_pytorch.bge import BGEAdapter
from cfg_pytorch.open_clip import OpenClipAdapter


# const
COND_DROP_KEY_NAME = 'cond_drop_prob'
TEXTS_KEY_NAME = 'texts'
TEXT_EMBEDS_KEY_NAME = 'text_embeds'
TEXT_CONDITIONER_NAME = 'text_conditioner'
CONDITION_FUNCTION_KEY_NAME = 'cond_fns'

TextCondReturn = namedtuple('TextCondReturn', [
    'embed',
    'mask'
])

def exists(val):
    return val is not None

def is_empty(l):
    return len(l) == 0
def default(*vals):
    for value in vals:
        if exists(value):
            return value
    return None

def cast_tuple(val, length=1):
    return val if isinstance(val, tuple) else ((val,) * length)

def pack_one(x,  pattern):
    return pack([x], pattern)

def unpack_one(x, ps , pattern):
    return unpack(x, ps , pattern)[0] # return the first element

def pack_one_with_inverse(x, pattern):
    packed, packed_shape = pack_one(x, pattern)
    def inverse(x, inverse_pattern = None):
        return unpack_one(x, packed_shape, default(inverse_pattern, pattern)) 
    return packed, inverse
# tensor helpers
def project(x, y):
    x, inverse = pack_one_with_inverse(x, 'b *') # b - batch
    y, _ = pack_one_with_inverse(y , 'b *')