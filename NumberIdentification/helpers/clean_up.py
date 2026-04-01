# Clean Up Part A
import gc
import torch

# 1) Move models off GPU (safe no-op on CPU)
def clean_up(objs, namespace=None):
    if namespace is None:
        namespace = globals()
    for name, value in list(namespace.items()):
        if any(value is obj for obj in objs):
            try:
                value.to('cpu')
            except Exception:
                pass
            namespace.pop(name, None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
