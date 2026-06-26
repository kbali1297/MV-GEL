import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_utils import cad_entity_to_mesh_faces

def _default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

data = json.loads(sys.stdin.read())
vertices = np.array(data["vertices"], dtype=np.float64)
faces = np.array(data["faces"], dtype=np.int64)
try:
    out = cad_entity_to_mesh_faces(
        data["cad_file_path"], vertices, faces,
        entity_type=data["feature"], entity_index=data["feature_idx"],
    )
    if isinstance(out, np.ndarray):
        out = out.tolist()
    elif isinstance(out, list):
        out = [list(e) if hasattr(e, "__iter__") else int(e) for e in out]
    sys.stdout.write("@@RESULT@@" + json.dumps({"ok": True, "result": out}, default=_default))
except Exception as e:
    sys.stdout.write("@@RESULT@@" + json.dumps({"ok": False, "err": repr(e)}))
