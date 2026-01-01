#!/usr/bin/env python3
"""
ONNX Export Script for SCNet

This script exports the SCNet model to ONNX format and validates the exported model
by comparing outputs between PyTorch and ONNX Runtime.

Usage:
    python -m scnet.export_onnx --checkpoint_path ./result/checkpoint.th --output_path ./scnet.onnx

Requirements:
    pip install onnx onnxruntime
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from ml_collections import ConfigDict

from .SCNet import SCNet
from .utils import load_model


def export_to_onnx(
    model: torch.nn.Module,
    output_path: str,
    sample_rate: int = 44100,
    audio_length_seconds: float = 11.0,
    opset_version: int = 17,
    dynamic_axes: bool = True,
    verbose: bool = True,
) -> str:
    """
    Export SCNet model to ONNX format.

    Args:
        model: The SCNet model to export
        output_path: Path to save the ONNX model
        sample_rate: Audio sample rate (default: 44100)
        audio_length_seconds: Length of audio in seconds for dummy input
        opset_version: ONNX opset version (default: 17)
        dynamic_axes: Whether to use dynamic axes for variable-length audio
        verbose: Print export information

    Returns:
        Path to the exported ONNX model
    """
    model.eval()
    device = next(model.parameters()).device

    # Create dummy input: (batch_size, audio_channels, audio_length)
    audio_length = int(sample_rate * audio_length_seconds)
    dummy_input = torch.randn(1, model.audio_channels, audio_length, device=device)

    if verbose:
        print(f"Exporting SCNet to ONNX...")
        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Output path: {output_path}")
        print(f"  Opset version: {opset_version}")

    # Define input/output names
    input_names = ["audio_input"]
    output_names = ["separated_sources"]

    # Define dynamic axes for variable-length audio
    if dynamic_axes:
        dynamic_axes_dict = {
            "audio_input": {0: "batch_size", 2: "audio_length"},
            "separated_sources": {0: "batch_size", 3: "audio_length"},
        }
    else:
        dynamic_axes_dict = None

    # Export to ONNX
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes_dict,
        verbose=False,
    )

    if verbose:
        print(f"  ONNX model exported successfully!")
        print(f"  Model size: {os.path.getsize(output_path) / (1024 * 1024):.2f} MB")

    return output_path


def validate_onnx_model(
    pytorch_model: torch.nn.Module,
    onnx_path: str,
    sample_rate: int = 44100,
    audio_length_seconds: float = 5.0,
    rtol: float = 1e-3,
    atol: float = 1e-5,
    verbose: bool = True,
) -> bool:
    """
    Validate ONNX model by comparing outputs with PyTorch model.

    Args:
        pytorch_model: The original PyTorch SCNet model
        onnx_path: Path to the exported ONNX model
        sample_rate: Audio sample rate
        audio_length_seconds: Length of test audio in seconds
        rtol: Relative tolerance for comparison
        atol: Absolute tolerance for comparison
        verbose: Print validation information

    Returns:
        True if validation passes, False otherwise
    """
    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        print("Error: onnx and onnxruntime are required for validation.")
        print("Install with: pip install onnx onnxruntime")
        return False

    if verbose:
        print("\nValidating ONNX model...")

    # Load and check ONNX model
    onnx_model = onnx.load(onnx_path)
    try:
        onnx.checker.check_model(onnx_model)
        if verbose:
            print("  ONNX model check passed!")
    except onnx.checker.ValidationError as e:
        print(f"  ONNX model check failed: {e}")
        return False

    # Create ONNX Runtime session
    providers = ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in ort.get_available_providers():
        providers.insert(0, "CUDAExecutionProvider")

    ort_session = ort.InferenceSession(onnx_path, providers=providers)
    if verbose:
        print(f"  Using providers: {ort_session.get_providers()}")

    # Create test input
    audio_length = int(sample_rate * audio_length_seconds)
    test_input = torch.randn(1, pytorch_model.audio_channels, audio_length)

    # Get PyTorch output
    pytorch_model.eval()
    pytorch_model.cpu()
    with torch.no_grad():
        pytorch_output = pytorch_model(test_input)

    # Get ONNX Runtime output
    ort_inputs = {"audio_input": test_input.numpy()}
    ort_output = ort_session.run(None, ort_inputs)[0]

    # Compare outputs
    pytorch_output_np = pytorch_output.numpy()

    if verbose:
        print(f"  PyTorch output shape: {pytorch_output_np.shape}")
        print(f"  ONNX output shape: {ort_output.shape}")
        print(f"  PyTorch output range: [{pytorch_output_np.min():.6f}, {pytorch_output_np.max():.6f}]")
        print(f"  ONNX output range: [{ort_output.min():.6f}, {ort_output.max():.6f}]")

    # Check if shapes match
    if pytorch_output_np.shape != ort_output.shape:
        print(f"  Shape mismatch: PyTorch {pytorch_output_np.shape} vs ONNX {ort_output.shape}")
        return False

    # Calculate differences
    abs_diff = np.abs(pytorch_output_np - ort_output)
    max_abs_diff = abs_diff.max()
    mean_abs_diff = abs_diff.mean()

    # Relative difference (avoiding division by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_diff = np.abs(pytorch_output_np - ort_output) / (np.abs(pytorch_output_np) + 1e-10)
        max_rel_diff = np.nanmax(rel_diff)
        mean_rel_diff = np.nanmean(rel_diff)

    if verbose:
        print(f"  Max absolute difference: {max_abs_diff:.6e}")
        print(f"  Mean absolute difference: {mean_abs_diff:.6e}")
        print(f"  Max relative difference: {max_rel_diff:.6e}")
        print(f"  Mean relative difference: {mean_rel_diff:.6e}")

    # Check if outputs are close enough
    is_close = np.allclose(pytorch_output_np, ort_output, rtol=rtol, atol=atol)

    if is_close:
        print("  Validation PASSED! ONNX model outputs match PyTorch model.")
    else:
        print(f"  Validation FAILED! Outputs differ beyond tolerance (rtol={rtol}, atol={atol})")
        # Still provide useful info even on failure
        looser_rtol = rtol * 10
        looser_atol = atol * 10
        is_loosely_close = np.allclose(pytorch_output_np, ort_output, rtol=looser_rtol, atol=looser_atol)
        if is_loosely_close:
            print(f"  Note: Validation would pass with looser tolerances (rtol={looser_rtol}, atol={looser_atol})")

    return is_close


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export SCNet model to ONNX format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to model checkpoint file (.th)",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="./conf/config.yaml",
        help="Path to model configuration file",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./exports/scnet.onnx",
        help="Output path for the ONNX model",
    )
    parser.add_argument(
        "--opset_version",
        type=int,
        default=17,
        help="ONNX opset version",
    )
    parser.add_argument(
        "--no_dynamic_axes",
        action="store_true",
        help="Disable dynamic axes (use fixed input shape)",
    )
    parser.add_argument(
        "--no_validate",
        action="store_true",
        help="Skip validation after export",
    )
    parser.add_argument(
        "--validation_length",
        type=float,
        default=5.0,
        help="Audio length in seconds for validation",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-3,
        help="Relative tolerance for validation",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-5,
        help="Absolute tolerance for validation",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load configuration
    config_path = Path(args.config_path)
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)

    with open(config_path, "r") as f:
        config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))

    # Create model
    print("Loading SCNet model...")
    model = SCNet(**config.model)
    model.eval()

    # Load checkpoint
    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        sys.exit(1)

    model = load_model(model, checkpoint_path)
    print(f"  Loaded checkpoint from {checkpoint_path}")

    # Export to ONNX
    output_path = export_to_onnx(
        model=model,
        output_path=args.output_path,
        sample_rate=config.data.samplerate,
        opset_version=args.opset_version,
        dynamic_axes=not args.no_dynamic_axes,
    )

    # Validate
    if not args.no_validate:
        success = validate_onnx_model(
            pytorch_model=model,
            onnx_path=output_path,
            sample_rate=config.data.samplerate,
            audio_length_seconds=args.validation_length,
            rtol=args.rtol,
            atol=args.atol,
        )
        if not success:
            print("\nWarning: Validation did not pass, but ONNX model was still exported.")
            print("The model may still work correctly for inference.")

    print(f"\nONNX export complete: {output_path}")


if __name__ == "__main__":
    main()
