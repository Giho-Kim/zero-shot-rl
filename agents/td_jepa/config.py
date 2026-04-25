"""Helpers for loading TD-JEPA defaults from the original project config."""

import ast
from pathlib import Path
from typing import Any, Dict

TRAIN_DEFAULTS = {
    "lr_predictor": 1e-4,
    "lr_phi": 1e-4,
    "lr_psi": 1e-4,
    "lr_actor": 1e-4,
    "weight_decay": 0.0,
    "train_goal_ratio": 0.5,
    "predictor_pessimism_penalty": 0.0,
    "actor_pessimism_penalty": 0.0,
    "stddev_clip": 0.3,
    "bc_coeff": 0.0,
    "log_eigvals": False,
    "scale_train_goals": False,
    "tilt_beta": 0.995,
    "tilt_temperature": 20.0,
    "tilt_temperature_start": 20.0,
    "tilt_temperature_end": 1.0,
    "tilt_candidate_multiplier": 2,
}

MODEL_DEFAULTS = {
    "actor_std": 0.2,
    "actor_use_full_encoder": True,
    "symmetric": False,
}

ARCH_DEFAULTS = {
    "norm_z": True,
    "rgb_encoder_name": "IdentityNN",
    "augmentator_name": "IdentityNN",
    "phi_predictor_embedding_layers": 2,
    "phi_predictor_num_parallel": 2,
    "psi_predictor_embedding_layers": 2,
    "psi_predictor_num_parallel": 2,
    "actor_embedding_layers": 2,
}

ZSRL_PROTOCOL_DEFAULTS = {
    "learning_steps": 1_000_000,
    "batch_size": 512,
    "eval_rollouts": 10,
    "eval_frequency": 20_000,
    "z_inference_steps": 10000,
    "compile": False,
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _launch_script_path(tilt: bool) -> Path:
    script_name = "launch_td_jepa_dmc_tilt.py" if tilt else "launch_td_jepa_dmc.py"
    return _repo_root() / "td_jepa" / "scripts" / "train" / "proprio" / script_name


def _load_base_cfg(script_path: Path) -> Dict[str, Any]:
    module = ast.parse(script_path.read_text())
    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "BASE_CFG":
                    return ast.literal_eval(node.value)
    raise RuntimeError(f"BASE_CFG not found in {script_path}")


def load_td_jepa_config(tilt: bool) -> Dict[str, Any]:
    """Load TD-JEPA defaults from the original project's launcher config."""

    base_cfg = _load_base_cfg(_launch_script_path(tilt))

    agent_cfg = base_cfg["agent"]
    model_cfg = agent_cfg["model"]
    arch_cfg = model_cfg["archi"]
    train_cfg = agent_cfg["train"]
    return {
        "name": "td_jepa",
        "seed": 42,
        "learning_steps": ZSRL_PROTOCOL_DEFAULTS["learning_steps"],
        "batch_size": ZSRL_PROTOCOL_DEFAULTS["batch_size"],
        "discount": train_cfg["discount"],
        "eval_rollouts": ZSRL_PROTOCOL_DEFAULTS["eval_rollouts"],
        "eval_frequency": ZSRL_PROTOCOL_DEFAULTS["eval_frequency"],
        "z_inference_steps": ZSRL_PROTOCOL_DEFAULTS["z_inference_steps"],
        "lr_predictor": train_cfg.get("lr_predictor", TRAIN_DEFAULTS["lr_predictor"]),
        "lr_phi": train_cfg.get("lr_phi", TRAIN_DEFAULTS["lr_phi"]),
        "lr_psi": train_cfg.get("lr_psi", TRAIN_DEFAULTS["lr_psi"]),
        "lr_actor": train_cfg.get("lr_actor", TRAIN_DEFAULTS["lr_actor"]),
        "weight_decay": train_cfg.get("weight_decay", TRAIN_DEFAULTS["weight_decay"]),
        "encoder_target_tau": train_cfg["encoder_target_tau"],
        "predictor_target_tau": train_cfg["predictor_target_tau"],
        "phi_ortho_coef": train_cfg["phi_ortho_coef"],
        "psi_ortho_coef": train_cfg["psi_ortho_coef"],
        "train_goal_ratio": train_cfg.get("train_goal_ratio", TRAIN_DEFAULTS["train_goal_ratio"]),
        "predictor_pessimism_penalty": train_cfg.get(
            "predictor_pessimism_penalty", TRAIN_DEFAULTS["predictor_pessimism_penalty"]
        ),
        "actor_pessimism_penalty": train_cfg.get(
            "actor_pessimism_penalty", TRAIN_DEFAULTS["actor_pessimism_penalty"]
        ),
        "stddev_clip": train_cfg.get("stddev_clip", TRAIN_DEFAULTS["stddev_clip"]),
        "bc_coeff": train_cfg.get("bc_coeff", TRAIN_DEFAULTS["bc_coeff"]),
        "log_eigvals": train_cfg.get("log_eigvals", TRAIN_DEFAULTS["log_eigvals"]),
        "scale_train_goals": train_cfg.get("scale_train_goals", TRAIN_DEFAULTS["scale_train_goals"]),
        "tilt": False,
        "tilt_beta": train_cfg.get("tilt_beta", TRAIN_DEFAULTS["tilt_beta"]),
        "tilt_temperature": train_cfg.get(
            "tilt_temperature", TRAIN_DEFAULTS["tilt_temperature"]
        ),
        "tilt_temperature_start": train_cfg.get(
            "tilt_temperature_start", train_cfg.get("tilt_temperature", TRAIN_DEFAULTS["tilt_temperature_start"])
        ),
        "tilt_temperature_end": train_cfg.get(
            "tilt_temperature_end", train_cfg.get("tilt_temperature", TRAIN_DEFAULTS["tilt_temperature_end"])
        ),
        "tilt_candidate_multiplier": TRAIN_DEFAULTS["tilt_candidate_multiplier"],
        "actor_std": model_cfg.get("actor_std", MODEL_DEFAULTS["actor_std"]),
        "actor_use_full_encoder": model_cfg.get(
            "actor_use_full_encoder", MODEL_DEFAULTS["actor_use_full_encoder"]
        ),
        "symmetric": model_cfg.get("symmetric", MODEL_DEFAULTS["symmetric"]),
        "compile": ZSRL_PROTOCOL_DEFAULTS["compile"],
        "phi_dim": arch_cfg["phi_dim"],
        "psi_dim": arch_cfg["psi_dim"],
        "norm_z": arch_cfg.get("norm_z", ARCH_DEFAULTS["norm_z"]),
        "rgb_encoder_name": arch_cfg.get("rgb_encoder", {}).get("name", ARCH_DEFAULTS["rgb_encoder_name"]),
        "augmentator_name": arch_cfg.get("augmentator", {}).get("name", ARCH_DEFAULTS["augmentator_name"]),
        "phi_predictor_hidden_dim": arch_cfg["phi_predictor"]["hidden_dim"],
        "phi_predictor_hidden_layers": arch_cfg["phi_predictor"]["hidden_layers"],
        "phi_predictor_embedding_layers": arch_cfg["phi_predictor"].get(
            "embedding_layers", ARCH_DEFAULTS["phi_predictor_embedding_layers"]
        ),
        "phi_predictor_num_parallel": arch_cfg["phi_predictor"].get(
            "num_parallel", ARCH_DEFAULTS["phi_predictor_num_parallel"]
        ),
        "psi_predictor_hidden_dim": arch_cfg["psi_predictor"]["hidden_dim"],
        "psi_predictor_hidden_layers": arch_cfg["psi_predictor"]["hidden_layers"],
        "psi_predictor_embedding_layers": arch_cfg["psi_predictor"].get(
            "embedding_layers", ARCH_DEFAULTS["psi_predictor_embedding_layers"]
        ),
        "psi_predictor_num_parallel": arch_cfg["psi_predictor"].get(
            "num_parallel", ARCH_DEFAULTS["psi_predictor_num_parallel"]
        ),
        "phi_mlp_hidden_dim": arch_cfg["phi_mlp_encoder"]["hidden_dim"],
        "phi_mlp_hidden_layers": arch_cfg["phi_mlp_encoder"]["hidden_layers"],
        "phi_mlp_norm": arch_cfg["phi_mlp_encoder"]["norm"],
        "psi_mlp_hidden_dim": arch_cfg["psi_mlp_encoder"]["hidden_dim"],
        "psi_mlp_hidden_layers": arch_cfg["psi_mlp_encoder"]["hidden_layers"],
        "psi_mlp_norm": arch_cfg["psi_mlp_encoder"]["norm"],
        "actor_hidden_dim": arch_cfg["actor"]["hidden_dim"],
        "actor_hidden_layers": arch_cfg["actor"]["hidden_layers"],
        "actor_embedding_layers": arch_cfg["actor"].get(
            "embedding_layers", ARCH_DEFAULTS["actor_embedding_layers"]
        ),
    }
