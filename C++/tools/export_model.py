"""One-time conversion of the existing WAFT checkpoint for native C++ inference."""
import argparse
from pathlib import Path
import sys
import math
import types

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / 'Python' / 'WATF_demo_project'))
from deploy.waft_deploy import build_model


@torch.jit.script
def dynamic_position(pos: torch.Tensor, tokens: torch.Tensor, image: torch.Tensor,
                     patch: int, offset: float, antialias: bool):
    height, width = image.size(2), image.size(3)
    count = pos.size(1) - 1
    if tokens.size(1) - 1 == count and height == width:
        return pos.to(tokens)
    side = int(math.sqrt(float(count)))
    h = float(height // patch) + offset
    w = float(width // patch) + offset
    grid = pos[:, 1:].float().reshape(1, side, side, tokens.size(2)).permute(0, 3, 1, 2)
    grid = torch.nn.functional.interpolate(grid, scale_factor=[h / side, w / side],
                                          mode='bicubic', antialias=antialias)
    return torch.cat([pos[:, :1].float(), grid.permute(0, 2, 3, 1).reshape(1, -1, tokens.size(2))], 1).to(tokens)


def prepare_dynamic_tokens(self, image, masks=None):
    tokens = self.patch_embed(image)
    if masks is not None:
        tokens = torch.where(masks.unsqueeze(-1), self.mask_token.to(tokens).unsqueeze(0), tokens)
    tokens = torch.cat([self.cls_token.expand(tokens.shape[0], -1, -1), tokens], 1)
    tokens = tokens + dynamic_position(self.pos_embed, tokens, image, self.patch_size,
                                      self.interpolate_offset, self.interpolate_antialias)
    if self.register_tokens is not None:
        tokens = torch.cat([tokens[:, :1], self.register_tokens.expand(tokens.shape[0], -1, -1), tokens[:, 1:]], 1)
    return tokens


class ExportModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, left, right):
        result = self.model({'img1': left, 'img2': right})
        return result['disp_pred'], result['delta_info_preds'][-1]


def normalized_identity(self, image):
    return image



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', type=int, default=960)
    parser.add_argument('--height', type=int, default=600)
    parser.add_argument('--checkpoint', type=Path, default=ROOT.parent / 'Python/WATF_demo_project/weights/DAv2B-4.pth')
    parser.add_argument('--config', type=Path, default=ROOT.parent / 'Python/WATF_demo_project/configs/SynLarge/DAv2B-4.yaml')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/model/waft_B.pt')
    parser.add_argument('--left', type=Path, default=ROOT / 'data/input/8/left.jpg')
    parser.add_argument('--right', type=Path, default=ROOT / 'data/input/8/right.jpg')
    args = parser.parse_args()
    if args.width <= 0 or args.height <= 0:
        parser.error('Input dimensions must be positive')
    if not torch.cuda.is_available():
        raise SystemExit('This export targets the existing CUDA C++ environment')
    device = 'cuda'
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch._C._set_graph_executor_optimize(False)
    model, _ = build_model(config_file=str(args.config), ckpt=str(args.checkpoint), device=device)
    wrapper = ExportModel(model).eval()
    inputs = []
    for path in [args.left, args.right]:
        image = cv2.imread(str(path))
        if image is None:
            raise SystemExit(f'Cannot read {path}')
        image = cv2.resize(image, (args.width, args.height))
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        inputs.append(torch.from_numpy(rgb).permute(2, 0, 1)[None].float().to(device))
    with torch.no_grad():
        reference = tuple(t.cpu() for t in wrapper(*inputs))
        original_normalize = model.normalize_image
        height, width = inputs[0].shape[-2:]
        pad_h, pad_w = (-height) % model.factor, (-width) % model.factor
        padding = [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
        normalized = [torch.nn.functional.pad(original_normalize(t), padding) for t in inputs]
        backbone = model.encoder.encoder.get_base_model()
        backbone.prepare_tokens_with_masks = types.MethodType(prepare_dynamic_tokens, backbone)
        model.normalize_image = types.MethodType(normalized_identity, model)
        core = torch.jit.trace(wrapper, tuple(normalized), check_trace=False)
        traced = core
        actual = tuple(t[..., pad_h // 2:pad_h // 2 + height, pad_w // 2:pad_w // 2 + width] for t in traced(*normalized))
    for expected, got in zip(reference, actual):
        np.testing.assert_allclose(got.float().cpu().numpy(), expected.float().cpu().numpy(),
                                   rtol=1e-4, atol=1e-4)
    path = args.output
    for width, height in [(640, 480), (800, 512), (777, 511), (592, 592)]:
        resized = [torch.nn.functional.interpolate(t, size=(height, width), mode='bilinear', align_corners=False) for t in inputs]
        backbone.prepare_tokens_with_masks = types.MethodType(type(backbone).prepare_tokens_with_masks, backbone)
        model.normalize_image = original_normalize
        with torch.no_grad():
            expected = tuple(t.cpu() for t in wrapper(*resized))
        backbone.prepare_tokens_with_masks = types.MethodType(prepare_dynamic_tokens, backbone)
        ph, pw = (-height) % model.factor, (-width) % model.factor
        pad = [pw // 2, pw - pw // 2, ph // 2, ph - ph // 2]
        normalized = [torch.nn.functional.pad(original_normalize(t), pad) for t in resized]
        with torch.no_grad():
            got = tuple(t[..., ph // 2:ph // 2 + height, pw // 2:pw // 2 + width] for t in traced(*normalized))
        for a, b in zip(expected, got):
            np.testing.assert_allclose(b.float().cpu().numpy(), a.float().numpy(), rtol=2e-4, atol=2e-4)
        print(f'Dynamic shape validated: {width}x{height}', flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(path), _extra_files={'contract.txt': 'normalized_rgb_symmetric_pad16_dynamic_v1'})
    print(f'TorchScript saved: {path}; dynamic spatial dimensions; matches PyTorch FP32')


if __name__ == '__main__':
    main()
