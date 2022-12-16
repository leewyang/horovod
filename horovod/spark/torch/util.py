# Copyright 2019 Uber Technologies, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import io
import platform
import sys

from horovod.runner.common.util import codec


def is_module_available(module_name):
    _is_module_available = is_module_available_fn()
    return _is_module_available(module_name)


def is_module_available_fn():
    def _is_module_available(module_name):
        if sys.version_info <= (3, 3):
            # python 3.0 to 3.3
            import pkgutil
            torch_loader = pkgutil.find_loader(module_name)
        elif sys.version_info >= (3, 4):
            # python 3.4 and above
            import importlib
            torch_loader = importlib.util.find_spec(module_name)
        else:
            raise RuntimeError('Unsupported version of Python: {}'.format(platform.python_version()))

        return torch_loader is not None

    return _is_module_available


def serialize_fn():
    is_module_available = is_module_available_fn()

    def _serialize(model):
        """Serialize model into byte array encoded into base 64."""
        if is_module_available('torch'):
            import torch
            sys.modules["torch._C._nn"] = torch.nn.functional

        if isinstance(model, torch.jit.ScriptModule):
            # If torch model is converted to torchScript
            model = save_into_bio(model, torch.jit.save)

        serialized_obj = codec.dumps_base64(model)
        return serialized_obj

    return _serialize


def deserialize_fn():
    is_module_available = is_module_available_fn()

    def _deserialize(model_bytes_base64):
        """Deserialize model from byte array encoded in base 64."""
        if is_module_available('torch'):
            import torch
            sys.modules["torch._C._nn"] = torch.nn.functional

        obj = codec.loads_base64(model_bytes_base64)

        if not isinstance(obj, torch.nn.Module):
            obj.seek(0)
            bio = io.BytesIO(obj.read())
            obj = torch.jit.load(bio)

        return obj

    return _deserialize


def save_into_bio_fn():
    def save_into_bio(obj, save_obj_fn):
        """Serialize object into byte array encoded into base 64."""
        bio = io.BytesIO()
        save_obj_fn(obj, bio)
        bio.seek(0)
        return bio

    return save_into_bio


def save_into_bio(obj, save_obj_fn):
    _save_into_bio = save_into_bio_fn()
    return _save_into_bio(obj, save_obj_fn)


def encode_optimizers(optimizers, model):
    """Returns optimizer classes and modified states, where the param ids are mapped to their
    absolute indices in the associated model parameters.

    Note: the optimizer_states produced by this method should only be consumed by the
    decode_optimizers method.
    """
    model_param_ids = {id(p):i for i, p in enumerate(model.parameters())}     # id(param) -> index
    optimizer_classes = [opt.__class__ for opt in optimizers]
    optimizer_states = [opt.state_dict() for opt in optimizers]

    # get optimizer 'params' with model-relative indices
    optimizer_params = []
    for opt in optimizers:
        opt_param_group = []
        for param_group in opt.param_groups:
            opt_param_ids = [model_param_ids[id(p)] for p in param_group['params']]
            opt_param_group.append(opt_param_ids)
        optimizer_params.append(opt_param_group)
    return optimizer_classes, optimizer_states, optimizer_params


def decode_optimizers(optimizer_classes, optimizer_states, optimizer_params, model):
    """Reconstructs optimizers from classes, state_dicts, and param ids relative to the given
    model instance."""
    model_params = dict(enumerate(model.parameters()))                        # index -> param
    optimizers = []
    for opt_cls, state, params in zip(optimizer_classes, optimizer_states, optimizer_params):
        param_groups = []
        for param_group in params:
            opt_params = [model_params[i] for i in param_group]
            param_groups.append({'params': opt_params})
        opt = opt_cls(param_groups, lr=1)
        opt.load_state_dict(state)
        optimizers.append(opt)
    return optimizers
