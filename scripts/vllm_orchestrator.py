import argparse
import os
import signal
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


def _require_requests():
    try:
        import requests  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "requests is required. Install with `pip install requests`."
        ) from exc


def _require_fire():
    try:
        import fire  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "fire is required. Install with `pip install fire`."
        ) from exc


@dataclass
class OrchestratorConfig:
    model: str
    num_instances: int = 1
    base_port: int = 8000
    base_group_port: int = 51216
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    dtype: str = "auto"
    enable_prefix_caching: Optional[bool] = None
    max_model_len: Optional[int] = None
    host: str = "0.0.0.0"
    revision: Optional[str] = None
    seed: Optional[int] = None
    # If provided, map instances to these GPU ids. Otherwise auto round-robin.
    visible_gpus: Optional[List[int]] = None


class VLLMOrchestrator:
    """Spawn and manage multiple vLLM server instances with health checks.

    This tool does not auto-start anything. Use the `start` command explicitly.
    """

    def __init__(self, **kwargs):
        self.cfg = OrchestratorConfig(**kwargs)
        self._procs: List[Tuple[int, int, int, object]] = []

    # -------- Planning helpers --------
    def _plan_instances(self) -> List[Tuple[int, int, int]]:
        num_instances = self.cfg.num_instances

        if self.cfg.visible_gpus is not None and len(self.cfg.visible_gpus) > 0:
            gpu_ids = self.cfg.visible_gpus
        else:
            # Lazy import torch only when needed
            import importlib
            torch = importlib.import_module("torch")
            gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
            if gpu_count == 0:
                raise RuntimeError(
                    "No CUDA GPUs detected. Launch vLLM servers only on GPU machines."
                )
            gpu_ids = list(range(gpu_count))

        plan: List[Tuple[int, int, int]] = []  # (instance_idx, cuda_id, port)
        for i in range(num_instances):
            cuda_id = gpu_ids[i % len(gpu_ids)]
            port = self.cfg.base_port + i
            group_port = self.cfg.base_group_port + i
            plan.append((cuda_id, port, group_port))
        return plan

    def plan(self) -> List[Dict[str, int]]:
        plan = self._plan_instances()
        return [
            {"cuda_id": cuda_id, "port": port, "group_port": group_port}
            for (cuda_id, port, group_port) in plan
        ]

    # -------- Lifecycle --------
    def start(self) -> List[Dict[str, int]]:
        """Start N servers. Requires explicit user invocation."""
        import subprocess
        _require_requests()
        plan = self._plan_instances()

        started: List[Dict[str, int]] = []
        for idx, (cuda_id, port, group_port) in enumerate(plan):
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = str(cuda_id)
            cmd = self._build_server_cmd(port=port)
            proc = subprocess.Popen(cmd, env=env)
            self._procs.append((cuda_id, port, group_port, proc))
            started.append({"cuda_id": cuda_id, "port": port, "group_port": group_port})
        return started

    def _build_server_cmd(self, *, port: int) -> List[str]:
        # Call the existing trainers.vllm_server entry with its dataclass CLI
        py = sys.executable
        args = [
            py,
            "-m",
            "trainers.vllm_server",
            f"--model={self.cfg.model}",
            f"--tensor_parallel_size={self.cfg.tensor_parallel_size}",
            f"--gpu_memory_utilization={self.cfg.gpu_memory_utilization}",
            f"--dtype={self.cfg.dtype}",
            f"--host={self.cfg.host}",
            f"--port={port}",
        ]
        if self.cfg.enable_prefix_caching is not None:
            args.append(f"--enable_prefix_caching={self.cfg.enable_prefix_caching}")
        if self.cfg.max_model_len is not None:
            args.append(f"--max_model_len={self.cfg.max_model_len}")
        if self.cfg.revision is not None:
            args.append(f"--revision={self.cfg.revision}")
        if self.cfg.seed is not None:
            args.append(f"--seed={self.cfg.seed}")
        return args

    def status(self, timeout: float = 0.5) -> List[Dict[str, object]]:
        _require_requests()
        import requests

        out: List[Dict[str, object]] = []
        for (cuda_id, port, group_port, proc) in self._procs:
            try:
                r = requests.get(f"http://{self.cfg.host}:{port}/health/", timeout=timeout)
                healthy = r.status_code == 200
            except Exception:
                healthy = False
            out.append(
                {
                    "cuda_id": cuda_id,
                    "port": port,
                    "group_port": group_port,
                    "pid": proc.pid if proc and proc.poll() is None else None,
                    "healthy": healthy,
                }
            )
        return out

    def stop(self, sig: int = signal.SIGTERM, wait_seconds: float = 10.0) -> int:
        """Stop all started servers."""
        stopped = 0
        for (_, _, _, proc) in self._procs:
            if proc and proc.poll() is None:
                try:
                    proc.send_signal(sig)
                except Exception:
                    pass
        # Wait a bit
        end = time.time() + wait_seconds
        for (_, _, _, proc) in self._procs:
            if proc and proc.poll() is None:
                remaining = max(0.0, end - time.time())
                try:
                    proc.wait(timeout=remaining)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
            if proc and proc.poll() is not None:
                stopped += 1
        self._procs.clear()
        return stopped


def _make_fire_cli():
    _require_fire()
    import fire

    def main(**kwargs):
        return VLLMOrchestrator(**kwargs)

    fire.Fire(main)


def _make_argparse_cli():
    parser = argparse.ArgumentParser(description="vLLM Orchestrator")
    parser.add_argument("--help-fire", action="store_true", help="Show Fire-based CLI help.")
    args, _ = parser.parse_known_args()
    if args.help_fire:
        _make_fire_cli()
    else:
        parser.print_help()


if __name__ == "__main__":
    # Prefer Fire CLI; if unavailable, print basic help via argparse without starting anything.
    try:
        _make_fire_cli()
    except ImportError:
        print(
            "Fire is not installed. Install it with `pip install fire`, or import the class "
            "`scripts.vllm_orchestrator.VLLMOrchestrator` and use it programmatically."
        )
        _make_argparse_cli()


