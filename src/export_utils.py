import os
from typing import Any, Optional
import torch as th
from stable_baselines3.common.callbacks import BaseCallback

class OnnxablePolicy(th.nn.Module):
    """
    Wrapper for a policy to make it exportable to ONNX/PyTorch.
    It performs rounding on actions to mimic environment discretization.
    """
    def __init__(self, actor: th.nn.Module, action_space=None, decimals=2):
        super().__init__()
        self.actor = actor
        self.decimals = decimals
        # action_space might be used for clipping/scaling if added later

    def forward(self, observation: th.Tensor) -> th.Tensor:
        # NOTE: Post-processing (clipping/unscaling actions) is NOT included by default.
        # SAC actor output is usually squash by tanh.
        # SB3 SAC actor returns (action, log_std) during forward or just action if deterministic.
        # For inference, we use deterministic=True.
        action = self.actor(observation, deterministic=True)
        rounded_action = th.round(action, decimals=self.decimals)
        
        return rounded_action

class ExportCallback(BaseCallback):
    """
    Callback for exporting the model to ONNX and PyTorch formats.
    Can be triggered by other callbacks (like AltitudeCurriculumCallback).
    """
    def __init__(self, model_dir: str, default_filename: str = "best_model", verbose: int = 0):
        super(ExportCallback, self).__init__(verbose)
        self.model_dir = model_dir
        self.default_filename = default_filename

    def trigger_export(self, filename: Optional[str] = None, model: Optional[Any] = None):
        """
        Exports the current model to ONNX and PyTorch.
        """
        export_model = model if model is not None else self.model
        if export_model is None:
            if self.verbose > 0:
                print("[Export] Error: No model provided for export.")
            return

        base_fname = filename if filename else self.default_filename
        # Remove extension if provided to handle both .onnx and .pt
        if base_fname.endswith(".onnx") or base_fname.endswith(".pt"):
            base_fname = os.path.splitext(base_fname)[0]

        onnx_path = os.path.join(self.model_dir, f"{base_fname}.onnx")
        torch_path = os.path.join(self.model_dir, f"{base_fname}.pt")
        
        if self.verbose > 0:
            print(f"Exporting model to ONNX: {onnx_path}")
            print(f"Exporting model to PyTorch: {torch_path}")
        
        # 1. Export to PyTorch (JIT Traced Actor policy)
        # Wrap the policy for export
        onnxable_model = OnnxablePolicy(export_model.policy.actor, export_model.action_space)
        
        # Define dummy input
        observation_size = export_model.observation_space.shape
        dummy_input = th.randn(1, *observation_size)
        
        # Trace and save
        traced_model = th.jit.trace(onnxable_model, dummy_input)
        th.jit.save(traced_model, torch_path)

        # 2. Export to ONNX
        # Export to ONNX
        th.onnx.export(
            onnxable_model,
            dummy_input,
            onnx_path,
            opset_version=15,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"}
            }
        )

    def _on_step(self) -> bool:
        """
        Called when a new best model is found (if used as callback_on_new_best).
        """
        self.trigger_export()
        return True
