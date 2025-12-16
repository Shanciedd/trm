from typing import Optional, Any, List, Dict, Tuple
from dataclasses import dataclass
import os
import json
import yaml
import shutil

import torch
from torch import nn

import hydra
import pydantic
from omegaconf import DictConfig

from evaluators.arc import ARC
from dataset.common import PuzzleDatasetMetadata
from utils.functions import load_model_class, get_model_source_path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LossConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")
    name: str


class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")
    name: str
    loss: LossConfig


class EvalConfig(pydantic.BaseModel):
    arch: ArchConfig
    data_paths: List[str]
    global_batch_size: int
    project_name: Optional[str] = None
    run_name: Optional[str] = None
    load_checkpoint: Optional[str] = None
    checkpoint_path: Optional[str] = None
    seed: int = 0
    max_puzzles: Optional[int] = None
    max_test_inputs_per_puzzle: Optional[int] = None


@dataclass
class EvalState:
    model: nn.Module
    carry: Any


def load_checkpoint(model: nn.Module, config: EvalConfig):
    state_dict = torch.load(config.load_checkpoint, map_location=device, weights_only=False)
    puzzle_emb_name = "_orig_mod.model.inner.puzzle_emb.weights"
    expected_shape = model.model.puzzle_emb.weights.shape  # type: ignore
    if puzzle_emb_name in state_dict:
        pe = state_dict[puzzle_emb_name]
        if pe.shape != expected_shape:
            state_dict[puzzle_emb_name] = torch.mean(pe, dim=0, keepdim=True).expand(expected_shape).contiguous()
    model.load_state_dict(state_dict, assign=True)


def create_model(config: EvalConfig, vocab_size: int, seq_len: int, num_puzzle_identifiers: int):
    model_cfg = dict(
        **config.arch.__pydantic_extra__,  # type: ignore
        batch_size=config.global_batch_size,
        vocab_size=vocab_size,
        seq_len=seq_len,
        num_puzzle_identifiers=num_puzzle_identifiers,
        causal=False,
    )
    model_cls = load_model_class(config.arch.name)
    loss_cls = load_model_class(config.arch.loss.name)
    model = model_cls(model_cfg)
    model = loss_cls(model, **config.arch.loss.__pydantic_extra__)  # type: ignore
    if "DISABLE_COMPILE" not in os.environ:
        model = torch.compile(model)  # type: ignore
    load_checkpoint(model, config)
    return model


def init_eval_state(config: EvalConfig, vocab_size: int, seq_len: int, num_puzzle_identifiers: int):
    model = create_model(config, vocab_size, seq_len, num_puzzle_identifiers)
    return EvalState(model=model, carry=None)


def save_code_and_config(config: EvalConfig):
    os.makedirs(config.checkpoint_path, exist_ok=True)
    for path in [get_model_source_path(config.arch.name), get_model_source_path(config.arch.loss.name)]:
        if path is not None:
            shutil.copy(path, os.path.join(config.checkpoint_path, os.path.basename(path)))
    with open(os.path.join(config.checkpoint_path, "eval_config.yaml"), "w") as f:
        yaml.dump(config.model_dump(), f)


def load_arc_test_data(data_path: str):
    with open(os.path.join(data_path, "test_puzzles.json")) as f:
        test_puzzles = json.load(f)
    names = sorted(test_puzzles.keys())
    return test_puzzles, names


def truncate_arc_puzzles(test_puzzles, names, max_puzzles):
    if max_puzzles is None:
        return test_puzzles, names
    names = names[:max_puzzles]
    return {k: test_puzzles[k] for k in names}, names


def build_arc_eval_metadata(test_puzzles, vocab_size, seq_len):
    n = len(test_puzzles)
    return PuzzleDatasetMetadata(
        seq_len=seq_len,
        vocab_size=vocab_size,
        pad_id=0,
        ignore_label_id=0,
        blank_identifier_id=0,
        num_puzzle_identifiers=n + 1,
        total_groups=n,
        mean_puzzle_examples=1.0,
        total_puzzles=n,
        sets=["all"],
    )


def arc_grid_to_model_batch(grid):
    arr = torch.tensor(grid, dtype=torch.long, device=device).flatten().unsqueeze(0)
    return {
        "inputs": arr,
        "labels": torch.zeros_like(arr),
        "puzzle_identifiers": torch.zeros(1, dtype=torch.long, device=device),
    }


def run_single_arc_inference(model, batch, return_keys):
    carry = model.initial_carry(batch)
    while True:
        carry, _, _, preds, done = model(carry=carry, batch=batch, return_keys=return_keys)
        if done:
            break
    return preds


def arc_style_evaluate(eval_state, evaluator, test_puzzles, max_test_inputs_per_puzzle):
    evaluator.begin_eval()
    return_keys = set(evaluator.required_outputs)
    for puzzle in test_puzzles.values():
        tests = puzzle["test"]
        if max_test_inputs_per_puzzle is not None:
            tests = tests[:max_test_inputs_per_puzzle]
        for pair in tests:
            batch = arc_grid_to_model_batch(pair["input"])
            preds = run_single_arc_inference(eval_state.model, batch, return_keys)
            evaluator.update_batch(batch, preds)


def finalize_arc_results(evaluator, save_path):
    return evaluator.result(save_path=save_path, rank=0, world_size=1, group=None)


@hydra.main(config_path="config", config_name="cfg_eval", version_base=None)
def launch(hydra_config: DictConfig):
    config = EvalConfig(**hydra_config)  # type: ignore

    if config.project_name is None:
        config.project_name = "ARC-eval"
    if config.run_name is None:
        config.run_name = "run"
    if config.checkpoint_path is None:
        config.checkpoint_path = os.path.join("eval_results", config.project_name, config.run_name)

    torch.manual_seed(config.seed)

    test_puzzles, names = load_arc_test_data(config.data_paths[0])
    test_puzzles, names = truncate_arc_puzzles(test_puzzles, names, config.max_puzzles)

    vocab_size = config.arch.__pydantic_extra__["vocab_size"]  # type: ignore
    seq_len = config.arch.__pydantic_extra__["seq_len"]        # type: ignore

    eval_metadata = build_arc_eval_metadata(test_puzzles, vocab_size, seq_len)
    eval_state = init_eval_state(config, vocab_size, seq_len, eval_metadata.num_puzzle_identifiers)
    eval_state.model.eval()

    evaluator = ARC(
        data_path=config.data_paths[0],
        eval_metadata=eval_metadata,
        submission_K=2,
    )

    save_code_and_config(config)

    arc_style_evaluate(
        eval_state,
        evaluator,
        test_puzzles,
        config.max_test_inputs_per_puzzle,
    )

    metrics = finalize_arc_results(evaluator, config.checkpoint_path)

    with open(os.path.join(config.checkpoint_path, "metrics.yaml"), "w") as f:
        yaml.dump(metrics, f)


if __name__ == "__main__":
    launch()
