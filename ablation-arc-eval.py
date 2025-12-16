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
    if config.load_checkpoint is None:
        raise ValueError("load_checkpoint is None")

    print(f"[INFO] Loading checkpoint: {config.load_checkpoint}")
    raw_state = torch.load(config.load_checkpoint, map_location=device, weights_only=False)

    state_dict = {}

    for k, v in raw_state.items():
        if k.startswith("_orig_mod."):
            k = k[len("_orig_mod."):]
        state_dict[k] = v

    puzzle_emb_name = "model.inner.puzzle_emb.weights"
    if puzzle_emb_name in state_dict:
        pe = state_dict[puzzle_emb_name]
        expected_shape = model.model.puzzle_emb.weights.shape  # type: ignore
        if pe.shape != expected_shape:
            print(
                f"[WARN] Resetting puzzle embedding "
                f"{pe.shape} -> {expected_shape}"
            )
            state_dict[puzzle_emb_name] = (
                torch.mean(pe, dim=0, keepdim=True)
                .expand(expected_shape)
                .contiguous()
            )

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    print("[INFO] Checkpoint loaded")
    if missing:
        print("[WARN] Missing keys:", missing)
    if unexpected:
        print("[WARN] Unexpected keys:", unexpected)


def create_model(
    config: EvalConfig,
    num_puzzle_identifiers: int,
):
    model_cfg = dict(
        **config.arch.__pydantic_extra__,
        batch_size=config.global_batch_size,
        num_puzzle_identifiers=num_puzzle_identifiers,
        causal=False,
    )

    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.arch.loss.name)

    model: nn.Module = model_cls(model_cfg)
    model = loss_head_cls(model, **config.arch.loss.__pydantic_extra__)  # type: ignore

    model = model.to(device)

    if not os.environ.get("ARC_EVAL_NO_COMPILE", "1") == "1":
        model = torch.compile(model)

    load_checkpoint(model, config)
    return model


def init_eval_state(
    config: EvalConfig,
    num_puzzle_identifiers: int,
):
    model = create_model(
        config=config,
        num_puzzle_identifiers=num_puzzle_identifiers,
    )
    return EvalState(model=model, carry=None)


def save_code_and_config(config: EvalConfig):
    os.makedirs(config.checkpoint_path, exist_ok=True)

    for path in [
        get_model_source_path(config.arch.name),
        get_model_source_path(config.arch.loss.name),
    ]:
        if path is not None:
            shutil.copy(path, os.path.join(config.checkpoint_path, os.path.basename(path)))

    with open(os.path.join(config.checkpoint_path, "eval_config.yaml"), "w") as f:
        yaml.dump(config.model_dump(), f)


def load_arc_test_data(data_path: str):
    with open(os.path.join(data_path, "test_puzzles.json"), "r") as f:
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


def arc_grid_to_model_batch(grid, puzzle_id):
    arr = torch.tensor(grid, dtype=torch.long, device=device).flatten().unsqueeze(0)

    return {
        "inputs": arr,
        "labels": torch.full_like(arr, -100),
        "puzzle_identifiers": torch.tensor([puzzle_id], dtype=torch.long, device=device),
    }


def run_single_arc_inference(model, batch, return_keys):
    carry = model.initial_carry(batch)

    while True:
        carry, _, _, preds, done = model(
            carry=carry,
            batch=batch,
            return_keys=return_keys,
        )
        if done:
            break

    return preds


def arc_style_evaluate(
    eval_state,
    evaluator,
    test_puzzles,
    max_test_inputs_per_puzzle,
):
    evaluator.begin_eval()
    return_keys = set(evaluator.required_outputs)

    for puzzle_idx, puzzle in enumerate(test_puzzles.values(), start=1):
        tests = puzzle["test"]

        if max_test_inputs_per_puzzle is not None:
            tests = tests[:max_test_inputs_per_puzzle]

        for pair in tests:
            batch = arc_grid_to_model_batch(pair["input"], puzzle_idx)
            preds = run_single_arc_inference(eval_state.model, batch, return_keys)
            evaluator.update_batch(batch, preds)


def finalize_arc_results(evaluator, save_path):
    return evaluator.result(
        save_path=save_path,
        rank=0,
        world_size=1,
        group=None,
    )


@hydra.main(config_path="config", config_name="cfg_eval", version_base=None)
def launch(hydra_config: DictConfig):
    config = EvalConfig(**hydra_config)  # type: ignore

    if config.project_name is None:
        config.project_name = "ARC-eval"
    if config.run_name is None:
        config.run_name = "run"
    if config.checkpoint_path is None:
        config.checkpoint_path = os.path.join(
            "eval_results",
            config.project_name,
            config.run_name,
        )

    os.makedirs(config.checkpoint_path, exist_ok=True)

    torch.manual_seed(config.seed)

    test_puzzles, names = load_arc_test_data(config.data_paths[0])
    test_puzzles, names = truncate_arc_puzzles(
        test_puzzles,
        names,
        config.max_puzzles,
    )

    vocab_size = config.arch.__pydantic_extra__["vocab_size"]
    seq_len = config.arch.__pydantic_extra__["seq_len"]

    eval_metadata = build_arc_eval_metadata(
        test_puzzles=test_puzzles,
        vocab_size=vocab_size,
        seq_len=seq_len,
    )

    eval_state = init_eval_state(
        config=config,
        num_puzzle_identifiers=eval_metadata.num_puzzle_identifiers,
    )
    eval_state.model.eval()

    evaluator = ARC(
        data_path=config.data_paths[0],
        eval_metadata=eval_metadata,
        submission_K=2,
    )

    save_code_and_config(config)

    arc_style_evaluate(
        eval_state=eval_state,
        evaluator=evaluator,
        test_puzzles=test_puzzles,
        max_test_inputs_per_puzzle=config.max_test_inputs_per_puzzle,
    )

    metrics = finalize_arc_results(
        evaluator=evaluator,
        save_path=config.checkpoint_path,
    )

    with open(os.path.join(config.checkpoint_path, "metrics.yaml"), "w") as f:
        yaml.dump(metrics, f)


if __name__ == "__main__":
    launch()