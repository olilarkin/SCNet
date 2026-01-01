#!/usr/bin/env python3
"""
CoreML Export Script for SCNet

This script exports the SCNet model to CoreML format and validates the exported model
by comparing outputs between PyTorch and CoreML.

Usage:
    python -m scnet.export_coreml --checkpoint_path ./result/checkpoint.th --output_path ./scnet.mlpackage

Requirements:
    pip install coremltools

Note: CoreML models can only run on macOS/iOS devices. Validation will be skipped
on non-Apple platforms, but the export will still work.
"""

import argparse
import os
import platform
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from ml_collections import ConfigDict

from .SCNet import SCNet
from .utils import load_model


def export_to_coreml(
    model: torch.nn.Module,
    output_path: str,
    sample_rate: int = 44100,
    audio_length_seconds: float = 11.0,
    compute_units: str = "ALL",
    minimum_deployment_target: str = "iOS15",
    convert_to_fp16: bool = False,
    verbose: bool = True,
) -> str:
    """
    Export SCNet model to CoreML format.

    Args:
        model: The SCNet model to export
        output_path: Path to save the CoreML model (.mlpackage or .mlmodel)
        sample_rate: Audio sample rate (default: 44100)
        audio_length_seconds: Length of audio in seconds for tracing
        compute_units: CoreML compute units ('ALL', 'CPU_ONLY', 'CPU_AND_GPU', 'CPU_AND_NE')
        minimum_deployment_target: Minimum iOS/macOS deployment target
        convert_to_fp16: Convert weights to float16 for smaller model size
        verbose: Print export information

    Returns:
        Path to the exported CoreML model
    """
    try:
        import coremltools as ct
    except ImportError:
        print("Error: coremltools is required for CoreML export.")
        print("Install with: pip install coremltools")
        sys.exit(1)

    model.eval()
    model.cpu()

    # Create dummy input for tracing: (batch_size, audio_channels, audio_length)
    audio_length = int(sample_rate * audio_length_seconds)
    dummy_input = torch.randn(1, model.audio_channels, audio_length)

    if verbose:
        print(f"Exporting SCNet to CoreML...")
        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Output path: {output_path}")
        print(f"  Compute units: {compute_units}")

    # Trace the model
    if verbose:
        print("  Tracing PyTorch model...")

    with torch.no_grad():
        traced_model = torch.jit.trace(model, dummy_input)

    # Define input shape with flexible sequence length
    # Note: CoreML supports flexible shapes with RangeDim
    input_shape = ct.Shape(
        shape=(
            1,  # batch_size (fixed to 1 for inference)
            model.audio_channels,
            ct.RangeDim(lower_bound=sample_rate, upper_bound=sample_rate * 60, default=audio_length),
        )
    )

    # Convert to CoreML
    if verbose:
        print("  Converting to CoreML...")

    # Map compute units string to enum
    compute_units_map = {
        "ALL": ct.ComputeUnit.ALL,
        "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
        "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
        "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
    }

    # Map deployment target
    deployment_target_map = {
        "iOS14": ct.target.iOS14,
        "iOS15": ct.target.iOS15,
        "iOS16": ct.target.iOS16,
        "iOS17": ct.target.iOS17,
        "macOS11": ct.target.macOS11,
        "macOS12": ct.target.macOS12,
        "macOS13": ct.target.macOS13,
        "macOS14": ct.target.macOS14,
    }

    try:
        deployment_target = deployment_target_map.get(minimum_deployment_target, ct.target.iOS15)
    except AttributeError:
        deployment_target = ct.target.iOS15

    try:
        mlmodel = ct.convert(
            traced_model,
            inputs=[ct.TensorType(name="audio_input", shape=input_shape)],
            outputs=[ct.TensorType(name="separated_sources")],
            compute_units=compute_units_map.get(compute_units, ct.ComputeUnit.ALL),
            minimum_deployment_target=deployment_target,
            convert_to=("mlprogram" if output_path.endswith(".mlpackage") else "neuralnetwork"),
        )
    except Exception as e:
        # Fallback: try with fixed shape if flexible shape fails
        if verbose:
            print(f"  Flexible shape conversion failed ({e}), trying fixed shape...")

        mlmodel = ct.convert(
            traced_model,
            inputs=[ct.TensorType(name="audio_input", shape=dummy_input.shape)],
            outputs=[ct.TensorType(name="separated_sources")],
            compute_units=compute_units_map.get(compute_units, ct.ComputeUnit.ALL),
            convert_to=("mlprogram" if output_path.endswith(".mlpackage") else "neuralnetwork"),
        )

    # Add metadata
    mlmodel.author = "SCNet"
    mlmodel.short_description = "SCNet: Sparse Compression Network for Music Source Separation"
    mlmodel.input_description["audio_input"] = "Stereo audio input (batch, channels, samples)"
    mlmodel.output_description["separated_sources"] = "Separated sources (batch, sources, channels, samples)"

    # Convert to FP16 if requested
    if convert_to_fp16:
        if verbose:
            print("  Converting to float16...")
        mlmodel = ct.models.neural_network.quantization_utils.quantize_weights(
            mlmodel, nbits=16
        )

    # Save the model
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    mlmodel.save(output_path)

    if verbose:
        # Calculate model size
        if output_path.endswith(".mlpackage"):
            total_size = sum(
                os.path.getsize(os.path.join(dirpath, filename))
                for dirpath, _, filenames in os.walk(output_path)
                for filename in filenames
            )
        else:
            total_size = os.path.getsize(output_path)

        print(f"  CoreML model exported successfully!")
        print(f"  Model size: {total_size / (1024 * 1024):.2f} MB")

    return output_path


def validate_coreml_model(
    pytorch_model: torch.nn.Module,
    coreml_path: str,
    sample_rate: int = 44100,
    audio_length_seconds: float = 5.0,
    rtol: float = 1e-2,
    atol: float = 1e-4,
    verbose: bool = True,
) -> bool:
    """
    Validate CoreML model by comparing outputs with PyTorch model.

    Note: This function only works on macOS as CoreML runtime is required.

    Args:
        pytorch_model: The original PyTorch SCNet model
        coreml_path: Path to the exported CoreML model
        sample_rate: Audio sample rate
        audio_length_seconds: Length of test audio in seconds
        rtol: Relative tolerance for comparison
        atol: Absolute tolerance for comparison
        verbose: Print validation information

    Returns:
        True if validation passes, False otherwise
    """
    # Check if running on macOS
    if platform.system() != "Darwin":
        if verbose:
            print("\nSkipping CoreML validation (not running on macOS)")
            print("CoreML models can only be validated on macOS/iOS devices.")
        return True  # Return True to not fail the export

    try:
        import coremltools as ct
    except ImportError:
        print("Error: coremltools is required for validation.")
        return False

    if verbose:
        print("\nValidating CoreML model...")

    # Load CoreML model
    try:
        mlmodel = ct.models.MLModel(coreml_path)
    except Exception as e:
        print(f"  Error loading CoreML model: {e}")
        return False

    # Create test input
    audio_length = int(sample_rate * audio_length_seconds)
    test_input = torch.randn(1, pytorch_model.audio_channels, audio_length)

    # Get PyTorch output
    pytorch_model.eval()
    pytorch_model.cpu()
    with torch.no_grad():
        pytorch_output = pytorch_model(test_input)

    # Get CoreML output
    try:
        coreml_input = {"audio_input": test_input.numpy()}
        coreml_output = mlmodel.predict(coreml_input)
        coreml_output_np = coreml_output["separated_sources"]
    except Exception as e:
        print(f"  Error running CoreML inference: {e}")
        return False

    # Compare outputs
    pytorch_output_np = pytorch_output.numpy()

    if verbose:
        print(f"  PyTorch output shape: {pytorch_output_np.shape}")
        print(f"  CoreML output shape: {coreml_output_np.shape}")
        print(f"  PyTorch output range: [{pytorch_output_np.min():.6f}, {pytorch_output_np.max():.6f}]")
        print(f"  CoreML output range: [{coreml_output_np.min():.6f}, {coreml_output_np.max():.6f}]")

    # Check if shapes match
    if pytorch_output_np.shape != coreml_output_np.shape:
        print(f"  Shape mismatch: PyTorch {pytorch_output_np.shape} vs CoreML {coreml_output_np.shape}")
        return False

    # Calculate differences
    abs_diff = np.abs(pytorch_output_np - coreml_output_np)
    max_abs_diff = abs_diff.max()
    mean_abs_diff = abs_diff.mean()

    # Relative difference (avoiding division by zero)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_diff = np.abs(pytorch_output_np - coreml_output_np) / (np.abs(pytorch_output_np) + 1e-10)
        max_rel_diff = np.nanmax(rel_diff)
        mean_rel_diff = np.nanmean(rel_diff)

    if verbose:
        print(f"  Max absolute difference: {max_abs_diff:.6e}")
        print(f"  Mean absolute difference: {mean_abs_diff:.6e}")
        print(f"  Max relative difference: {max_rel_diff:.6e}")
        print(f"  Mean relative difference: {mean_rel_diff:.6e}")

    # Check if outputs are close enough
    # Note: CoreML may have larger numerical differences due to different floating point handling
    is_close = np.allclose(pytorch_output_np, coreml_output_np, rtol=rtol, atol=atol)

    if is_close:
        print("  Validation PASSED! CoreML model outputs match PyTorch model.")
    else:
        print(f"  Validation FAILED! Outputs differ beyond tolerance (rtol={rtol}, atol={atol})")
        # Check with looser tolerances
        looser_rtol = rtol * 10
        looser_atol = atol * 10
        is_loosely_close = np.allclose(pytorch_output_np, coreml_output_np, rtol=looser_rtol, atol=looser_atol)
        if is_loosely_close:
            print(f"  Note: Validation would pass with looser tolerances (rtol={looser_rtol}, atol={looser_atol})")

    return is_close


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export SCNet model to CoreML format",
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
        default="./exports/scnet.mlpackage",
        help="Output path for the CoreML model (.mlpackage or .mlmodel)",
    )
    parser.add_argument(
        "--compute_units",
        type=str,
        choices=["ALL", "CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE"],
        default="ALL",
        help="CoreML compute units to use",
    )
    parser.add_argument(
        "--minimum_deployment_target",
        type=str,
        default="iOS15",
        help="Minimum deployment target (e.g., iOS15, iOS16, macOS12)",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Convert weights to float16 for smaller model size",
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
        default=1e-2,
        help="Relative tolerance for validation (CoreML typically needs looser tolerances)",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-4,
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

    # Export to CoreML
    output_path = export_to_coreml(
        model=model,
        output_path=args.output_path,
        sample_rate=config.data.samplerate,
        compute_units=args.compute_units,
        minimum_deployment_target=args.minimum_deployment_target,
        convert_to_fp16=args.fp16,
    )

    # Validate
    if not args.no_validate:
        success = validate_coreml_model(
            pytorch_model=model,
            coreml_path=output_path,
            sample_rate=config.data.samplerate,
            audio_length_seconds=args.validation_length,
            rtol=args.rtol,
            atol=args.atol,
        )
        if not success:
            print("\nWarning: Validation did not pass, but CoreML model was still exported.")
            print("The model may still work correctly for inference.")

    print(f"\nCoreML export complete: {output_path}")


if __name__ == "__main__":
    main()
