import os
from typing import Any, Optional
import torch as th
import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback

class SACOnnxablePolicy(th.nn.Module):
    """
    Wrapper for a policy to make it exportable to ONNX/PyTorch.
    SAC Actor returns a single action tensor.
    """
    def __init__(self, actor: th.nn.Module, action_space: gym.spaces.Box, decimals=2):
        super().__init__()
        self.actor = actor
        self.decimals = decimals

    def forward(self, observation: th.Tensor) -> th.Tensor:
        action = self.actor(observation, deterministic=True)
        rounded_action = th.round(action, decimals=self.decimals)       
        return rounded_action

class PPOOnnxablePolicy(th.nn.Module):
    """
    Wrapper for a policy to make it exportable to ONNX/PyTorch.
    PPO ActorCriticPolicy returns (action, value, log_prob).
    """
    def __init__(self, policy: th.nn.Module, action_space: gym.spaces.Box, decimals=2):
        super().__init__()
        self.policy = policy
        self.decimals = decimals

    def forward(self, observation: th.Tensor) -> th.Tensor:
        # PPO returns (action, value, log_prob)
        action, _, _ = self.policy(observation, deterministic=True)
        rounded_action = th.round(action, decimals=self.decimals)       
        return rounded_action

class SACExportCallback(BaseCallback):
    """
    Callback for exporting the SAC model to ONNX and PyTorch formats.
    """
    def __init__(self, model_dir: str, default_filename: str = "best_model", verbose: int = 0):
        super(SACExportCallback, self).__init__(verbose)
        self.model_dir = model_dir
        self.default_filename = default_filename

    def trigger_export(self, filename: Optional[str] = None, model: Optional[Any] = None):
        """
        Exports the current model to ONNX and PyTorch.
        """
        export_model = model if model is not None else self.model
        if export_model is None:
            if self.verbose > 0:
                print("[SACExport] Error: No model provided for export.")
            return

        base_fname = filename if filename else self.default_filename
        # Remove extension if provided to handle both .onnx and .pt
        if base_fname.endswith(".onnx") or base_fname.endswith(".pt"):
            base_fname = os.path.splitext(base_fname)[0]

        onnx_path = os.path.join(self.model_dir, f"{base_fname}_sac.onnx")
        torch_path = os.path.join(self.model_dir, f"{base_fname}_sac.pt")
        tflite_path = os.path.join(self.model_dir, f"{base_fname}_sac.tflite")
        zip_path = os.path.join(self.model_dir, f"{base_fname}_sac.zip")
        
        if self.verbose > 0:
            print(f"Exporting SAC model to ONNX: {onnx_path}")
            print(f"Exporting SAC model to PyTorch: {torch_path}")
            print(f"Exporting SAC model to TFLite: {tflite_path}")
            print(f"Saving SAC model to: {zip_path}")
        
        # Save the whole model
        export_model.save(zip_path)

        # SAC: Wrap the actor policy for export
        onnxable_model = SACOnnxablePolicy(export_model.policy.actor, export_model.action_space)
        
        # Define dummy input
        observation_size = export_model.observation_space.shape
        dummy_input = th.randn(1, *observation_size)
        
        # Trace and save PyTorch model
        traced_model = th.jit.trace(onnxable_model, dummy_input)
        th.jit.save(traced_model, torch_path)

        # Export to TFLite
        try:
            import litert_torch
            from torch._export.converter import TS2EPConverter
            
            # Use TS2EPConverter to bridge TorchScript to ExportedProgram for LiteRT
            converter = TS2EPConverter(traced_model, (dummy_input,))
            exported_program = converter.convert()
            edge_model = litert_torch.convert(exported_program.module(), (dummy_input,))
            edge_model.export(tflite_path)
        except Exception as e:
            if self.verbose > 0:
                print(f"[SACExport] Error exporting to TFLite: {e}")

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
        return True

class PPOExportCallback(BaseCallback):
    """
    Callback for exporting the PPO model to ONNX and PyTorch formats.
    """
    def __init__(self, model_dir: str, default_filename: str = "best_model_ppo", verbose: int = 0):
        super(PPOExportCallback, self).__init__(verbose)
        self.model_dir = model_dir
        self.default_filename = default_filename

    def trigger_export(self, filename: Optional[str] = None, model: Optional[Any] = None):
        """
        Exports the current model to ONNX and PyTorch.
        """
        export_model = model if model is not None else self.model
        if export_model is None:
            if self.verbose > 0:
                print("[PPOExport] Error: No model provided for export.")
            return

        base_fname = filename if filename else self.default_filename
        if base_fname.endswith(".onnx") or base_fname.endswith(".pt"):
            base_fname = os.path.splitext(base_fname)[0]

        onnx_path = os.path.join(self.model_dir, f"{base_fname}_ppo.onnx")
        torch_path = os.path.join(self.model_dir, f"{base_fname}_ppo.pt")
        tflite_path = os.path.join(self.model_dir, f"{base_fname}_ppo.tflite")
        zip_path = os.path.join(self.model_dir, f"{base_fname}_ppo.zip")
        
        if self.verbose > 0:
            print(f"Exporting PPO model to ONNX: {onnx_path}")
            print(f"Exporting PPO model to PyTorch: {torch_path}")
            print(f"Exporting PPO model to TFLite: {tflite_path}")
            print(f"Saving PPO model to: {zip_path}")
        
        # Save the whole model
        export_model.save(zip_path)

        # PPO: Wrap the full policy
        onnxable_model = PPOOnnxablePolicy(export_model.policy, export_model.action_space)
        
        observation_size = export_model.observation_space.shape
        dummy_input = th.randn(1, *observation_size)
        
        # Trace and save PyTorch model
        traced_model = th.jit.trace(onnxable_model, dummy_input)
        th.jit.save(traced_model, torch_path)

        # Export to TFLite
        try:
            import litert_torch
            from torch._export.converter import TS2EPConverter
            
            # Use TS2EPConverter to bridge TorchScript to ExportedProgram for LiteRT
            converter = TS2EPConverter(traced_model, (dummy_input,))
            exported_program = converter.convert()
            edge_model = litert_torch.convert(exported_program.module(), (dummy_input,))
            edge_model.export(tflite_path)
        except Exception as e:
            if self.verbose > 0:
                print(f"[PPOExport] Error exporting to TFLite: {e}")

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
        self.trigger_export()
        return True
