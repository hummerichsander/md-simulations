"""Re-export a legacy CGSchNet/mlcg checkpoint so the current stack can load it.

The Clementi-group transferable checkpoints were pickled as full module graphs
against a **pre-2.5 torch-geometric**, whose ``MessagePassing`` stored an
``Inspector`` at ``torch_geometric.nn.conv.utils.inspector`` with internals that
no longer exist (the class moved to ``torch_geometric.inspector`` and was
rewritten in PyG 2.5). Such a checkpoint cannot be unpickled under the PyG that
``mlcg`` now requires (>=2.6.1).

This tool loads the legacy checkpoint by (1) registering a permissive stand-in at
the old module path so unpickling can materialise the dead ``Inspector`` objects,
and (2) rebuilding a *current* PyG inspector for every ``MessagePassing`` module
instead of trusting the pickled one. The repaired model is re-saved with
current-PyG internals and afterwards loads with a plain ``torch.load`` — no
scaffolding needed. Only the framework plumbing is touched; parameters, buffers
and submodules are preserved untouched (verify checksums are printed)."""

import argparse
import sys
import types

import torch


def _install_legacy_pyg_shims() -> None:
    """Make a pre-2.5 checkpoint unpickle under current torch-geometric.

    Registers a stand-in ``Inspector`` at the removed module path and patches
    ``MessagePassing.__setstate__`` to rebuild a current inspector per module."""
    from torch_geometric.nn import MessagePassing

    shim = types.ModuleType("torch_geometric.nn.conv.utils.inspector")

    class _DeadInspector:
        """Absorbs the pickled pre-2.5 inspector state; never actually used."""

        def __setstate__(self, state: dict) -> None:
            self.__dict__.update(state or {})

        def __getattr__(self, name: str):
            return None

    shim.Inspector = _DeadInspector
    sys.modules.setdefault(
        "torch_geometric.nn.conv.utils",
        types.ModuleType("torch_geometric.nn.conv.utils"),
    )
    sys.modules["torch_geometric.nn.conv.utils.inspector"] = shim

    def _rebuilt_setstate(self, data: dict) -> None:
        # Materialise the pickled module, then re-run the *current*
        # MessagePassing.__init__ so every framework attribute this PyG expects
        # (inspector, hooks, decomposed_layers, explain flags, aggr_module, ...)
        # is installed. __init__ resets the nn.Module containers, so preserve and
        # restore parameters/buffers/submodules and any subclass attributes.
        torch.nn.Module.__setstate__(self, data)
        saved = dict(self.__dict__)
        MessagePassing.__init__(
            self,
            aggr=data.get("aggr", "add") or "add",
            flow=data.get("flow", "source_to_target"),
            node_dim=data.get("node_dim", -2),
        )
        self._parameters = saved["_parameters"]
        self._buffers = saved["_buffers"]
        self._modules = saved["_modules"]
        for key, value in saved.items():
            if not hasattr(self, key):
                setattr(self, key, value)

    MessagePassing.__setstate__ = _rebuilt_setstate


def reexport(model_file: str, out_file: str) -> None:
    """Load a legacy checkpoint via shims and re-save it for the current stack.

    :param model_file: Path to the legacy (pre-2.5-PyG) ``.pt`` checkpoint.
    :param out_file: Path to write the re-exported, natively-loadable checkpoint."""
    _install_legacy_pyg_shims()
    from torch_geometric.nn import MessagePassing

    obj = torch.load(model_file, map_location="cpu", weights_only=False)
    model = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    n_mp = sum(isinstance(m, MessagePassing) for m in model.modules())
    n_params = sum(p.numel() for p in model.parameters())
    checksum = float(sum(p.detach().double().abs().sum() for p in model.parameters()))
    print(f"Loaded {type(model).__name__}: {n_params} params, {n_mp} conv modules")
    print(f"Parameter checksum (preserved across re-export): {checksum:.6f}")

    torch.save(model, out_file)
    print(f"Re-exported → {out_file}")


def main() -> None:
    """Entry point: re-export a legacy CGSchNet/mlcg checkpoint."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("model_file", help="Legacy (pre-2.5-PyG) checkpoint .pt file.")
    parser.add_argument(
        "-o", "--out", required=True, help="Output path for the re-exported checkpoint."
    )
    args = parser.parse_args()
    reexport(args.model_file, args.out)


if __name__ == "__main__":
    main()
