#!/usr/bin/env python3
"""
Validation Script for Exported SCNet Models

This script validates exported ONNX and CoreML models by comparing their outputs
with the original PyTorch model. It can also run inference benchmarks.

Usage:
    # Validate ONNX model
    python -m scnet.validate_exports --checkpoint_path ./result/checkpoint.th --onnx_path ./exports/scnet.onnx

    # Validate CoreML model (macOS only)
    python -m scnet.validate_exports --checkpoint_path ./result/checkpoint.th --coreml_path ./exports/scnet.mlpackage

    # Validate both models
    python -m scnet.validate_exports --checkpoint_path ./result/checkpoint.th \
        --onnx_path ./exports/scnet.onnx --coreml_path ./exports/scnet.mlpackage

    # Validate with real audio file
    python -m scnet.validate_exports --checkpoint_path ./result/checkpoint.th \
        --onnx_path ./exports/scnet.onnx --audio_file ./test_audio.wav

Requirements:
    pip install onnx onnxruntime  # For ONNX validation
    pip install coremltools       # For CoreML validation (macOS only)
    pip install soundfile         # For audio file testing
"""

import argparse
import os
import platform
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from ml_collections import ConfigDict

from .SCNet import SCNet
from .utils import load_model


def load_audio_file(file_path: str, sample_rate: int = 44100, channels: int = 2) -> torch.Tensor:
    """Load audio file and convert to the expected format."""
    try:
        import soundfile as sf
        import julius
    except ImportError:
        print("Error: soundfile and julius are required for audio file loading.")
        print("Install with: pip install soundfile julius")
        sys.exit(1)

    audio, sr = sf.read(file_path, dtype="float32")

    # Convert to torch tensor and ensure correct shape
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=0)  # Mono to stereo
    else:
        audio = audio.T  # (samples, channels) -> (channels, samples)

    audio = torch.from_numpy(audio)

    # Ensure stereo
    if audio.shape[0] == 1:
        audio = audio.repeat(2, 1)
    elif audio.shape[0] > 2:
        audio = audio[:2]

    # Resample if needed
    if sr != sample_rate:
        audio = julius.resample_frac(audio, sr, sample_rate)

    # Ensure correct number of channels
    if audio.shape[0] != channels:
        if channels == 1:
            audio = audio.mean(0, keepdim=True)
        else:
            audio = audio[:channels]

    return audio.unsqueeze(0)  # Add batch dimension


def run_pytorch_inference(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    device: str = "cpu",
) -> Tuple[np.ndarray, float]:
    """Run inference with PyTorch model and return output and time."""
    model.eval()
    model.to(device)
    input_tensor = input_tensor.to(device)

    # Warm-up run
    with torch.no_grad():
        _ = model(input_tensor)

    # Timed run
    if device == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()
    with torch.no_grad():
        output = model(input_tensor)

    if device == "cuda":
        torch.cuda.synchronize()

    elapsed_time = time.perf_counter() - start_time

    return output.cpu().numpy(), elapsed_time


def run_onnx_inference(
    onnx_path: str,
    input_tensor: torch.Tensor,
    use_gpu: bool = False,
) -> Tuple[np.ndarray, float]:
    """Run inference with ONNX Runtime and return output and time."""
    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError("onnxruntime is required for ONNX inference.")

    providers = ["CPUExecutionProvider"]
    if use_gpu and "CUDAExecutionProvider" in ort.get_available_providers():
        providers.insert(0, "CUDAExecutionProvider")

    session = ort.InferenceSession(onnx_path, providers=providers)

    input_np = input_tensor.numpy()
    ort_inputs = {"audio_input": input_np}

    # Warm-up run
    _ = session.run(None, ort_inputs)

    # Timed run
    start_time = time.perf_counter()
    output = session.run(None, ort_inputs)[0]
    elapsed_time = time.perf_counter() - start_time

    return output, elapsed_time


def run_coreml_inference(
    coreml_path: str,
    input_tensor: torch.Tensor,
) -> Tuple[np.ndarray, float]:
    """Run inference with CoreML and return output and time."""
    if platform.system() != "Darwin":
        raise RuntimeError("CoreML inference is only supported on macOS.")

    try:
        import coremltools as ct
    except ImportError:
        raise ImportError("coremltools is required for CoreML inference.")

    mlmodel = ct.models.MLModel(coreml_path)
    input_np = input_tensor.numpy()
    coreml_input = {"audio_input": input_np}

    # Warm-up run
    _ = mlmodel.predict(coreml_input)

    # Timed run
    start_time = time.perf_counter()
    output = mlmodel.predict(coreml_input)
    elapsed_time = time.perf_counter() - start_time

    return output["separated_sources"], elapsed_time


def compare_outputs(
    reference: np.ndarray,
    test: np.ndarray,
    name: str,
    rtol: float = 1e-3,
    atol: float = 1e-5,
    verbose: bool = True,
) -> Dict[str, float]:
    """Compare two outputs and return statistics."""
    abs_diff = np.abs(reference - test)
    max_abs_diff = float(abs_diff.max())
    mean_abs_diff = float(abs_diff.mean())

    with np.errstate(divide="ignore", invalid="ignore"):
        rel_diff = np.abs(reference - test) / (np.abs(reference) + 1e-10)
        max_rel_diff = float(np.nanmax(rel_diff))
        mean_rel_diff = float(np.nanmean(rel_diff))

    is_close = np.allclose(reference, test, rtol=rtol, atol=atol)

    # Signal-to-noise ratio
    signal_power = np.mean(reference**2)
    noise_power = np.mean((reference - test) ** 2)
    snr = 10 * np.log10(signal_power / (noise_power + 1e-10))

    if verbose:
        print(f"\n{name} Comparison:")
        print(f"  Shape: reference={reference.shape}, test={test.shape}")
        print(f"  Reference range: [{reference.min():.6f}, {reference.max():.6f}]")
        print(f"  Test range: [{test.min():.6f}, {test.max():.6f}]")
        print(f"  Max absolute diff: {max_abs_diff:.6e}")
        print(f"  Mean absolute diff: {mean_abs_diff:.6e}")
        print(f"  Max relative diff: {max_rel_diff:.6e}")
        print(f"  Mean relative diff: {mean_rel_diff:.6e}")
        print(f"  SNR: {snr:.2f} dB")
        print(f"  Validation (rtol={rtol}, atol={atol}): {'PASSED' if is_close else 'FAILED'}")

    return {
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "max_rel_diff": max_rel_diff,
        "mean_rel_diff": mean_rel_diff,
        "snr_db": snr,
        "passed": is_close,
    }


def benchmark(
    pytorch_model: Optional[torch.nn.Module],
    onnx_path: Optional[str],
    coreml_path: Optional[str],
    input_tensor: torch.Tensor,
    num_runs: int = 5,
    verbose: bool = True,
) -> Dict[str, Dict[str, float]]:
    """Run benchmarks for all available models."""
    results = {}

    # Get audio duration for calculating real-time factor
    sample_rate = 44100  # Assuming 44.1kHz
    audio_duration = input_tensor.shape[-1] / sample_rate

    if verbose:
        print(f"\nBenchmarking ({num_runs} runs each)...")
        print(f"  Audio duration: {audio_duration:.2f} seconds")
        print(f"  Input shape: {input_tensor.shape}")

    # PyTorch CPU
    if pytorch_model is not None:
        times = []
        for _ in range(num_runs):
            _, elapsed = run_pytorch_inference(pytorch_model, input_tensor.clone(), "cpu")
            times.append(elapsed)

        avg_time = np.mean(times)
        std_time = np.std(times)
        rtf = avg_time / audio_duration

        results["pytorch_cpu"] = {
            "avg_time": avg_time,
            "std_time": std_time,
            "rtf": rtf,
        }

        if verbose:
            print(f"  PyTorch (CPU): {avg_time*1000:.1f}ms ± {std_time*1000:.1f}ms (RTF: {rtf:.3f}x)")

        # PyTorch GPU if available
        if torch.cuda.is_available():
            pytorch_model.cuda()
            times = []
            for _ in range(num_runs):
                _, elapsed = run_pytorch_inference(
                    pytorch_model, input_tensor.clone(), "cuda"
                )
                times.append(elapsed)

            avg_time = np.mean(times)
            std_time = np.std(times)
            rtf = avg_time / audio_duration

            results["pytorch_cuda"] = {
                "avg_time": avg_time,
                "std_time": std_time,
                "rtf": rtf,
            }

            if verbose:
                print(f"  PyTorch (CUDA): {avg_time*1000:.1f}ms ± {std_time*1000:.1f}ms (RTF: {rtf:.3f}x)")

            pytorch_model.cpu()

    # ONNX
    if onnx_path is not None:
        times = []
        for _ in range(num_runs):
            _, elapsed = run_onnx_inference(onnx_path, input_tensor.clone(), use_gpu=False)
            times.append(elapsed)

        avg_time = np.mean(times)
        std_time = np.std(times)
        rtf = avg_time / audio_duration

        results["onnx_cpu"] = {
            "avg_time": avg_time,
            "std_time": std_time,
            "rtf": rtf,
        }

        if verbose:
            print(f"  ONNX (CPU): {avg_time*1000:.1f}ms ± {std_time*1000:.1f}ms (RTF: {rtf:.3f}x)")

        # ONNX GPU
        try:
            import onnxruntime as ort
            if "CUDAExecutionProvider" in ort.get_available_providers():
                times = []
                for _ in range(num_runs):
                    _, elapsed = run_onnx_inference(onnx_path, input_tensor.clone(), use_gpu=True)
                    times.append(elapsed)

                avg_time = np.mean(times)
                std_time = np.std(times)
                rtf = avg_time / audio_duration

                results["onnx_cuda"] = {
                    "avg_time": avg_time,
                    "std_time": std_time,
                    "rtf": rtf,
                }

                if verbose:
                    print(f"  ONNX (CUDA): {avg_time*1000:.1f}ms ± {std_time*1000:.1f}ms (RTF: {rtf:.3f}x)")
        except ImportError:
            pass

    # CoreML (macOS only)
    if coreml_path is not None and platform.system() == "Darwin":
        times = []
        for _ in range(num_runs):
            _, elapsed = run_coreml_inference(coreml_path, input_tensor.clone())
            times.append(elapsed)

        avg_time = np.mean(times)
        std_time = np.std(times)
        rtf = avg_time / audio_duration

        results["coreml"] = {
            "avg_time": avg_time,
            "std_time": std_time,
            "rtf": rtf,
        }

        if verbose:
            print(f"  CoreML: {avg_time*1000:.1f}ms ± {std_time*1000:.1f}ms (RTF: {rtf:.3f}x)")

    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate exported SCNet models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to PyTorch model checkpoint file (.th)",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="./conf/config.yaml",
        help="Path to model configuration file",
    )
    parser.add_argument(
        "--onnx_path",
        type=str,
        default=None,
        help="Path to ONNX model file (.onnx)",
    )
    parser.add_argument(
        "--coreml_path",
        type=str,
        default=None,
        help="Path to CoreML model file (.mlpackage or .mlmodel)",
    )
    parser.add_argument(
        "--audio_file",
        type=str,
        default=None,
        help="Path to audio file for testing (optional)",
    )
    parser.add_argument(
        "--audio_length",
        type=float,
        default=5.0,
        help="Audio length in seconds for random input testing",
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
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run inference benchmarks",
    )
    parser.add_argument(
        "--benchmark_runs",
        type=int,
        default=5,
        help="Number of benchmark runs",
    )
    parser.add_argument(
        "--save_outputs",
        action="store_true",
        help="Save outputs as numpy files for debugging",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./validation_outputs",
        help="Directory to save output files",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.onnx_path is None and args.coreml_path is None:
        print("Error: At least one of --onnx_path or --coreml_path must be specified.")
        sys.exit(1)

    # Load configuration
    config_path = Path(args.config_path)
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)

    with open(config_path, "r") as f:
        config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))

    sample_rate = config.data.samplerate

    # Create and load PyTorch model
    print("Loading PyTorch model...")
    pytorch_model = SCNet(**config.model)
    pytorch_model.eval()

    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        sys.exit(1)

    pytorch_model = load_model(pytorch_model, checkpoint_path)
    print(f"  Loaded checkpoint from {checkpoint_path}")

    # Prepare input
    if args.audio_file is not None:
        print(f"\nLoading audio file: {args.audio_file}")
        input_tensor = load_audio_file(args.audio_file, sample_rate, pytorch_model.audio_channels)
        print(f"  Audio shape: {input_tensor.shape}")
    else:
        audio_length = int(sample_rate * args.audio_length)
        input_tensor = torch.randn(1, pytorch_model.audio_channels, audio_length)
        print(f"\nUsing random input: {input_tensor.shape}")

    # Get PyTorch reference output
    print("\nRunning PyTorch inference...")
    pytorch_output, pytorch_time = run_pytorch_inference(pytorch_model, input_tensor.clone())
    print(f"  Output shape: {pytorch_output.shape}")
    print(f"  Inference time: {pytorch_time*1000:.1f}ms")

    all_passed = True
    comparison_results = {}

    # Validate ONNX model
    if args.onnx_path is not None:
        print(f"\n{'='*50}")
        print("ONNX Validation")
        print("=" * 50)

        onnx_path = Path(args.onnx_path)
        if not onnx_path.exists():
            print(f"Error: ONNX model not found at {onnx_path}")
        else:
            try:
                import onnx

                # Check ONNX model
                onnx_model = onnx.load(str(onnx_path))
                onnx.checker.check_model(onnx_model)
                print("ONNX model check: PASSED")

                # Run inference
                print("Running ONNX inference...")
                onnx_output, onnx_time = run_onnx_inference(str(onnx_path), input_tensor.clone())
                print(f"  Output shape: {onnx_output.shape}")
                print(f"  Inference time: {onnx_time*1000:.1f}ms")

                # Compare outputs
                result = compare_outputs(
                    pytorch_output,
                    onnx_output,
                    "PyTorch vs ONNX",
                    rtol=args.rtol,
                    atol=args.atol,
                )
                comparison_results["onnx"] = result
                if not result["passed"]:
                    all_passed = False

            except Exception as e:
                print(f"ONNX validation error: {e}")
                all_passed = False

    # Validate CoreML model
    if args.coreml_path is not None:
        print(f"\n{'='*50}")
        print("CoreML Validation")
        print("=" * 50)

        coreml_path = Path(args.coreml_path)
        if not coreml_path.exists():
            print(f"Error: CoreML model not found at {coreml_path}")
        elif platform.system() != "Darwin":
            print("Skipping CoreML validation (not running on macOS)")
        else:
            try:
                print("Running CoreML inference...")
                coreml_output, coreml_time = run_coreml_inference(
                    str(coreml_path), input_tensor.clone()
                )
                print(f"  Output shape: {coreml_output.shape}")
                print(f"  Inference time: {coreml_time*1000:.1f}ms")

                # Use looser tolerances for CoreML
                coreml_rtol = args.rtol * 10
                coreml_atol = args.atol * 10

                # Compare outputs
                result = compare_outputs(
                    pytorch_output,
                    coreml_output,
                    "PyTorch vs CoreML",
                    rtol=coreml_rtol,
                    atol=coreml_atol,
                )
                comparison_results["coreml"] = result
                if not result["passed"]:
                    all_passed = False

            except Exception as e:
                print(f"CoreML validation error: {e}")
                all_passed = False

    # Run benchmarks if requested
    if args.benchmark:
        print(f"\n{'='*50}")
        print("Benchmarks")
        print("=" * 50)
        benchmark_results = benchmark(
            pytorch_model=pytorch_model,
            onnx_path=args.onnx_path if args.onnx_path and Path(args.onnx_path).exists() else None,
            coreml_path=args.coreml_path if args.coreml_path and Path(args.coreml_path).exists() else None,
            input_tensor=input_tensor,
            num_runs=args.benchmark_runs,
        )

    # Save outputs if requested
    if args.save_outputs:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"\nSaving outputs to {args.output_dir}/")

        np.save(os.path.join(args.output_dir, "pytorch_output.npy"), pytorch_output)
        print(f"  Saved pytorch_output.npy")

        if args.onnx_path and "onnx_output" in dir():
            np.save(os.path.join(args.output_dir, "onnx_output.npy"), onnx_output)
            print(f"  Saved onnx_output.npy")

        if args.coreml_path and "coreml_output" in dir():
            np.save(os.path.join(args.output_dir, "coreml_output.npy"), coreml_output)
            print(f"  Saved coreml_output.npy")

    # Summary
    print(f"\n{'='*50}")
    print("Summary")
    print("=" * 50)

    for name, result in comparison_results.items():
        status = "PASSED" if result["passed"] else "FAILED"
        print(f"  {name.upper()}: {status} (SNR: {result['snr_db']:.2f} dB)")

    if all_passed:
        print("\nAll validations PASSED!")
        return 0
    else:
        print("\nSome validations FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
