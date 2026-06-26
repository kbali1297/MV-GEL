---
license: mit
pipeline_tag: other
---

# PatchAlign3D: Local Feature Alignment for Dense 3D Shape Understanding

PatchAlign3D is an encoder-only 3D model that produces language-aligned patch-level features directly from point clouds. It enables zero-shot 3D part segmentation with fast single-pass inference without requiring test-time multi-view rendering.

- **Paper:** [PatchAlign3D: Local Feature Alignment for Dense 3D Shape understanding](https://huggingface.co/papers/2601.02457)
- **Project Page:** [https://souhail-hadgi.github.io/patchalign3dsite](https://souhail-hadgi.github.io/patchalign3dsite)
- **Repository:** [https://github.com/souhail-hadgi/PatchAlign3D](https://github.com/souhail-hadgi/PatchAlign3D)

## Sample Usage

You can run inference on a single shape and save per-point predictions using the following command from the official repository:

```bash
python patchalign3d/inference/infer.py \
  --ckpt /path/to/stage2_last.pt \
  --input /path/to/shape.npz \
  --labels "seat,back,leg,arm"
```

## Citation

```bibtex
@misc{hadgi2026patchalign3dlocalfeaturealignment,
  title={PatchAlign3D: Local Feature Alignment for Dense 3D Shape understanding},
  author={Souhail Hadgi and Bingchen Gong and Ramana Sundararaman and Emery Pierson and Lei Li and Peter Wonka and Maks Ovsjanikov},
  year={2026},
  eprint={2601.02457},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2601.02457},
}
```