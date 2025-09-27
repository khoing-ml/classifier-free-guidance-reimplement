from __future__ import annotations
from collections import namedtuple
from functools import wraps
from torch import nn, einsum, Tensor
from torch.nn import Module
from torch.nn import functional as F
from einops import rearrange, repeat, pack, unpack
import torch
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

    dtype = x.dtype  
    # cast to real type as F.normalize only supports float and double
    x, y = x.double(), y.double()
    #normalize 
    unit = F.normalize(y, dim=-1) ## normalize the value dim
    parallel = (x * unit).sum(dim = - 1, keepdim = True) * unit 
    orthogonal = x - parallel 
    return inverse(parallel.type(dtype)), inverse(orthogonal.type(dtype))

def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob
    
@typecheck
def classifier_free_guidance(
    fn: Callable,
    cond_drop_prob_keyname = COND_DROP_KEY_NAME,
    texts_key_name = TEXTS_KEY_NAME,
    text_embeds_key_name = TEXT_EMBEDS_KEY_NAME,
    cond_fns_keyname = CONDITION_FUNCTION_KEY_NAME,
    text_conditioner_name = TEXT_CONDITIONER_NAME
):
    fn_params = signature(fn).parameters
    auto_handle_text_condition = texts_key_name in fn_params or text_embeds_key_name in fn_params

    @wraps(fn)
    def inner(
        self,
        *args,
        cond_scale: float = 1.,
        rescale_phi: float = 0.,
        return_unconditioned: bool = False,
        remove_parallel_component: bool = False,
        keep_parallel_frac: float = 0., # in paper, they show complete removal to the best
        cfg_routed_kwargs: Dict[str, Tuple[Any, Any]] = dict(),   # to pass in separate arguments to forward and nulled forward calls (for handling caching when using CFG on transformer decoding)
        **kwargs
    ):
        @wraps(fn)
        def fn_maybe_with_text(self, *args, **kwargs):
            if auto_handle_text_condition:
                texts = kwargs.pop('texts', None)
                text_embeds = kwargs.pop('text_embeds', None)
                assert not (exists(texts) and exists(text_embeds)), "should only pass in raw texts or text embeddings, not both"

                raw_text_conditioner = cond_fns = None # raw text condition
                text_conditioner = getattr(self,text_conditioner_name, None) #text used for condition: \mathbf{c}
                cond_drop_prob = kwargs.pop(cond_drop_prob_keyname, None) #  ( w)
                assert not exists(cond_drop_prob) or 0. <= cond_drop_prob <= 1., f"{cond_drop_prob_keyname} should be between 0 and 1"
                if exists(texts) ^ exists(text_embeds):
                    assert is_bearable(texts, List[str] | None), f'keywords `{texts_key_name}` should be List[str]'
                    assert exists(text_conditioner) and is_bearable(text_conditioner, Conditioner), f'if passing in `{texts_key_name}` or `{text_embeds_key_name}`, the model should have a `{text_conditioner_name}` attribute that is callable'
                    text_condition_input = dict(texts = texts) if exists(texts) else dict(text_embeds = text_embeds) 
                    cond_fns , raw_text_conditioner = text_conditioner(**text_condition_input, cond_drop_prob = cond_drop_prob)
                elif isinstance(text_conditioner, NullConditioner):
                    assert cond_drop_prob == 0., 'nothing to dropout'
                    cond_dns, raw_text_conditioner = text_conditioner()

                if 'cond_fns' in fn_params:
                    kwargs.update(cond_fns = cond_fns)
                
                if 'raw_text_conditioner' in fn_params:
                    kwargs.update(raw_text_conditioner = raw_text_conditioner)
            return  fn(self, *args, **kwargs)
        
        # main classifier free guidance logic
                
        
        if self.training:
            assert cond_scale == 1., 'you cannot do condition scaling when in traing mode'
            return fn_maybe_with_text(self, *args, **kwargs)
        assert cond_scale >= 1, 'invalid conditioning scale'

        kwargs_without_cond_dropout = {**kwargs, cond_drop_prob_keyname: 0.}
        kwargs_with_cond_dropout = {**kwargs, cond_drop_prob_keyname: 1.}

        fn_kwargs = {k : v[0] for k, v in cfg_routed_kwargs.items()}
        null_fn_kwargs = {k : v[1] for k, v in cfg_routed_kwargs.items()}

        outputs = fn_maybe_with_text(self, *args, **fn_kwargs, **kwargs_without_cond_dropout)

        if cond_scale == 1:
            return outputs

        logits, *rest = cast_tuple(outputs)

        null_outputs =  fn_maybe_with_text(self, *args, **null_fn_kwargs, **kwargs_with_cond_dropout)
        null_logits, *null_rest = cast_tuple(null_outputs)
        zipped_rest = tuple(zip(rest, null_rest))

        update = logits - null_logits
        if remove_parallel_component:




class Conditioner(Module):
    pass

class Identity(Module):
    def forward(self, t, *args, **kwargs):
        return t
   
class NullConditioner(Conditioner):
    @typecheck
    def __init__(self, *, hidden_dims: Tuple[int, ...], **kwargs):
        super().__init__()
        num_null_conditioners = len(hidden_dims)
        self.cond_fns = tuple(Identity() for _ in range(num_null_conditioners))
        self.register_buffer('_device_param', torch.tensor(0), persistant= False)

    @property
    def device(self):
        return next(self.buffers()).device
    
    @typecheck
    def embed_texts(self, texts : List[str]):
        assert False, 'null conditioner cannot embed text'
    
    def forward(self, *args, **kwargs):
        return self.cond_fns, None
    
