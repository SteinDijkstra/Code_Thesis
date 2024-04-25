import yaml
from typing import List, Optional
from pydantic import BaseModel, StrictStr
from pathlib import Path


class Inputlayer(BaseModel):
    number_input: int


class Layer(BaseModel):
    nodes: int
    activation: str
    kernel_size: Optional[int] = None


class Outputlayer(BaseModel):
    number_output: int
    type_output: Optional[str]


class Compilation(BaseModel):
    loss_function: str
    optimizer: str


class Training(BaseModel):
    batch_size: int
    epochs: int
    use_callback: bool


class Outputoptions(BaseModel):
    save_output: bool
    output_location: Path


class Variant(BaseModel):
    dropoutrate: Optional[float] = None
    tau: Optional[float] = None
    n_ensemble: Optional[int] = None
    r: Optional[float] = None
    n_draws: Optional[int] = None
    max_layers: Optional[int] = None
    min_nodes: Optional[int] = None
    max_nodes: Optional[int] = None
    activation_functions: Optional[List[str]] = None
    batchsizes: Optional[List[int]] = None
    optimizers: Optional[List[str]] = None


class Pedestrian(BaseModel):
    lookback: Optional[int] = None
    forward: Optional[int] = None
    nodes: Optional[int] = None
    activation: Optional[str] = None


class Options(BaseModel):
    inputlayer: Inputlayer
    layers: List[Layer]
    outputlayer: Outputlayer
    compilation: Compilation
    training: Training
    outputoptions: Outputoptions
    variant: Variant
    pedestrian: Pedestrian


def read_yaml(file_path: str) -> dict:
    with open(file_path, "r") as stream:
        config = yaml.safe_load(stream)
    return Options(**config)
