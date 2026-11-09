"""Prueba aislada de TabPFN fuera de Jupyter.
Uso (en Anaconda Prompt, con el entorno TesisCONDA activado):
    python -X faulthandler tabpfn_test.py cuda
    python -X faulthandler tabpfn_test.py cpu
"""
import os, sys, faulthandler
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
faulthandler.enable()
import numpy as np, torch
from sklearn.datasets import make_classification

device = sys.argv[1] if len(sys.argv) > 1 else "cuda"
print("torch", torch.__version__, "| cuda build", torch.version.cuda, "| disponible", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0), "| arch list:", torch.cuda.get_arch_list())
    x = torch.randn(1000, 1000, device="cuda"); print("matmul cuda OK:", (x @ x).sum().item() != 0, flush=True)

import tabpfn
from tabpfn import TabPFNClassifier
print("tabpfn", getattr(tabpfn, "__version__", "?"), flush=True)

X, y = make_classification(n_samples=270, n_features=18, n_informative=10, n_classes=3, random_state=0)
X = X.astype(np.float32)
clf = TabPFNClassifier(device=device)
print("fit...", flush=True);     clf.fit(X[:200], y[:200])
print("predict...", flush=True); p = clf.predict_proba(X[200:])
print("OK", p.shape)
