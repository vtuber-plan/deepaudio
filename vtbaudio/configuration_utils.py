


import copy
import json
import os
from typing import Any, Dict, Union
from .utils import PushToHubMixin, logging
from . import __version__

logger = logging.get_logger(__name__)

class AudioModelConfig(PushToHubMixin):
    model_type: str = ""
    attribute_map: Dict[str, str] = {}

    def __setattr__(self, key: str, value: Any):
        if key in super().__getattribute__("attribute_map"):
            key = super().__getattribute__("attribute_map")[key]
        super().__setattr__(key, value)

    def __getattribute__(self, key: str):
        if key != "attribute_map" and key in super().__getattribute__("attribute_map"):
            key = super().__getattribute__("attribute_map")[key]
        return super().__getattribute__(key)

    def __init__(self, **kwargs):
        self.sampling_rate = kwargs.pop("sampling_rate", 48000)

        self.filter_length = kwargs.pop("filter_length", 2048)
        self.hop_length = kwargs.pop("hop_length", 512)
        self.win_length = kwargs.pop("win_length", 2048)

        # Additional attributes without default values
        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                logger.error(f"Can't set {key} with value {value} for {self}")
                raise err

    def dict_torch_dtype_to_str(self, d: Dict[str, Any]) -> None:
        if d.get("torch_dtype", None) is not None and not isinstance(d["torch_dtype"], str):
            d["torch_dtype"] = str(d["torch_dtype"]).split(".")[1]
        for value in d.values():
            if isinstance(value, dict):
                self.dict_torch_dtype_to_str(value)

    def to_dict(self) -> Dict[str, Any]:
        output = copy.deepcopy(self.__dict__)
        if hasattr(self.__class__, "model_type"):
            output["model_type"] = self.__class__.model_type

        # vtbaudio version when serializing the model
        output["vtbaudio_version"] = __version__

        self.dict_torch_dtype_to_str(output)

        return output

    def to_json_string(self) -> str:
        config_dict = self.to_dict()
        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path: Union[str, os.PathLike], use_diff: bool = True):
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string(use_diff=use_diff))

    def __repr__(self):
        return f"{self.__class__.__name__} {self.to_json_string()}"

class DiscriminatorConfig(AudioModelConfig):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

class VocoderConfig(AudioModelConfig):
    def __init__(self, **kwargs) -> None:
        self.n_mel_channels = kwargs.pop("n_mel_channels", 80)
        self.mel_fmin = kwargs.pop("mel_fmin", 0.0)
        self.mel_fmax = kwargs.pop("mel_fmax", None)

        super().__init__(**kwargs)