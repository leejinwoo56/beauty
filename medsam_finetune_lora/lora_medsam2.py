"""
lora_medsam2.py
---------------
MedSAM2 이미지 인코더(Hiera backbone)에 LoRA를 적용하는 모듈.

구조:
  - LoRAQKV  : MultiScaleAttention.qkv (fused Linear) 를 감싸서
                Q 와 V 부분에만 LoRA delta 를 더함 (K 는 원본 유지)
  - apply_lora_to_model : 모델 전체를 순회하며 모든 qkv 레이어를 교체
  - get_trainable_params: 학습할 파라미터(LoRA + 선택적으로 mask decoder) 반환

사용 예:
    from lora_medsam2 import apply_lora_to_model, get_trainable_params
    apply_lora_to_model(model, r=4, lora_alpha=4)
    params = get_trainable_params(model, train_decoder=True)
    optimizer = torch.optim.AdamW(params, lr=1e-4)
"""

import math
import torch
import torch.nn as nn


# ──────────────────────────────────────────────────────────────────────────────
# LoRA 모듈
# ──────────────────────────────────────────────────────────────────────────────

class LoRAQKV(nn.Module):
    """
    MultiScaleAttention.qkv 를 대체하는 LoRA 래퍼.

    원본 qkv : nn.Linear(dim, dim_out * 3)   — frozen
    LoRA delta: Q 와 V 에만 적용
        lora_q : dim → r → dim_out
        lora_v : dim → r → dim_out

    forward 입력  : x  [..., dim]
    forward 출력  : [..., dim_out * 3]   (Q+ΔQ, K, V+ΔV) concatenated
    """

    def __init__(self, original_linear: nn.Linear, r: int, lora_alpha: float):
        super().__init__()

        in_dim = original_linear.in_features          # dim
        out_dim = original_linear.out_features // 3   # dim_out (Q, K, V 각각)

        # 원본 가중치 freeze
        self.original = original_linear
        for p in self.original.parameters():
            p.requires_grad = False

        # LoRA 파라미터 (bias 없음)
        self.lora_q_A = nn.Linear(in_dim, r, bias=False)
        self.lora_q_B = nn.Linear(r, out_dim, bias=False)
        self.lora_v_A = nn.Linear(in_dim, r, bias=False)
        self.lora_v_B = nn.Linear(r, out_dim, bias=False)

        self.scale = lora_alpha / r

        # 초기화: A ~ Kaiming uniform, B = 0 (처음엔 delta = 0)
        nn.init.kaiming_uniform_(self.lora_q_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_q_B.weight)
        nn.init.kaiming_uniform_(self.lora_v_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_v_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 원본 QKV
        qkv = self.original(x)              # [..., dim_out * 3]
        d = qkv.shape[-1] // 3             # dim_out

        q_orig = qkv[..., :d]
        k      = qkv[..., d : 2 * d]
        v_orig = qkv[..., 2 * d :]

        # LoRA delta
        q = q_orig + self.lora_q_B(self.lora_q_A(x)) * self.scale
        v = v_orig + self.lora_v_B(self.lora_v_A(x)) * self.scale

        return torch.cat([q, k, v], dim=-1)


# ──────────────────────────────────────────────────────────────────────────────
# 모델에 LoRA 적용
# ──────────────────────────────────────────────────────────────────────────────

def apply_lora_to_model(
    model: nn.Module,
    r: int = 4,
    lora_alpha: float = 4.0,
) -> int:
    """
    모델 내 모든 MultiScaleAttention.qkv 레이어를 LoRAQKV 로 교체.

    - LoRA 파라미터를 제외한 모든 파라미터는 freeze
    - mask decoder (sam_mask_decoder) 는 이 함수에서 freeze 하지 않음
      → get_trainable_params 에서 선택적으로 학습 가능

    Args:
        model      : SAM2Base 인스턴스 (build_sam2 로 생성)
        r          : LoRA rank
        lora_alpha : LoRA scaling factor (보통 r 과 동일하게 설정)

    Returns:
        교체된 qkv 레이어 수
    """
    # 1) 전체 파라미터 freeze
    for p in model.parameters():
        p.requires_grad = False

    # 2) image_encoder 내 qkv 레이어 찾아서 교체
    replaced = 0
    image_encoder = model.image_encoder  # ImageEncoder (trunk = Hiera)

    for module_name, module in image_encoder.named_modules():
        # MultiScaleAttention 인스턴스인지 확인
        # hieradet.py: class MultiScaleAttention
        cls_name = type(module).__name__
        if cls_name == "MultiScaleAttention":
            original_qkv = module.qkv  # nn.Linear(dim, dim_out * 3)
            device = next(original_qkv.parameters()).device
            module.qkv = LoRAQKV(original_qkv, r=r, lora_alpha=lora_alpha).to(device)
            replaced += 1

    print(f"[LoRA] {replaced}개의 MultiScaleAttention.qkv 레이어에 LoRA 적용 완료")
    print(f"       rank={r}, alpha={lora_alpha}")
    return replaced


# ──────────────────────────────────────────────────────────────────────────────
# SAMed-style 기본 프롬프트 임베딩 추가
# ──────────────────────────────────────────────────────────────────────────────

def add_default_prompt(model: nn.Module, n_tokens: int = 2) -> nn.Parameter:
    """
    SAMed 방식의 학습 가능한 기본 프롬프트 임베딩을 모델에 추가.

    prompt encoder 를 우회하고 mask decoder 에 직접 주입되는
    task-specific sparse embedding.  학습을 통해 "SMAS를 찾아라"는
    신호를 인코딩하게 됨.

    apply_lora_to_model() 호출 이후에 실행해야 함.
    (전체 freeze 이후 새 Parameter 를 추가하므로 requires_grad=True 유지)

    Args:
        model   : apply_lora_to_model 이 호출된 SAM2Base 인스턴스
        n_tokens: 학습할 프롬프트 토큰 수 (SAMed 논문 기본값 = 4,
                  가벼운 실험에는 2 권장)

    Returns:
        nn.Parameter [1, n_tokens, 256]
    """
    embed_dim = model.hidden_dim  # 256
    prompt = nn.Parameter(torch.zeros(1, n_tokens, embed_dim))
    nn.init.trunc_normal_(prompt, std=0.02)
    model.default_prompt_embedding = prompt  # model에 등록 → named_parameters()에 포함
    print(f"[LoRA] SAMed 기본 프롬프트 추가: {n_tokens}개 토큰 × {embed_dim}dim")
    return prompt


# ──────────────────────────────────────────────────────────────────────────────
# 학습 파라미터 수집
# ──────────────────────────────────────────────────────────────────────────────

def get_trainable_params(
    model: nn.Module,
    train_decoder: bool = True,
) -> list:
    """
    학습할 파라미터 그룹 반환.

    Args:
        model        : apply_lora_to_model 이 호출된 SAM2Base 인스턴스
        train_decoder: True 이면 sam_mask_decoder 파라미터도 학습

    Returns:
        optimizer 에 넘길 param_groups 리스트
    """
    lora_params = []
    decoder_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            # mask decoder 는 여기서 따로 unfreeze
            if train_decoder and "sam_mask_decoder" in name:
                param.requires_grad = True
                decoder_params.append(param)
            continue
        lora_params.append(param)

    total = sum(p.numel() for p in lora_params + decoder_params)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"[LoRA] 학습 파라미터: {total:,} / {all_params:,} "
          f"({100 * total / all_params:.2f}%)")

    param_groups = [{"params": lora_params, "lr": 1e-4}]
    if decoder_params:
        param_groups.append({"params": decoder_params, "lr": 1e-4})

    return param_groups


# ──────────────────────────────────────────────────────────────────────────────
# 체크포인트 저장 / 로드 (LoRA 파라미터만)
# ──────────────────────────────────────────────────────────────────────────────

def save_lora_checkpoint(model: nn.Module, path: str):
    """LoRA 파라미터 + mask decoder 파라미터만 저장."""
    state = {
        name: param
        for name, param in model.state_dict().items()
        if ("lora_" in name) or ("sam_mask_decoder" in name)
        or ("default_prompt_embedding" in name)
    }
    torch.save(state, path)
    print(f"[LoRA] 체크포인트 저장: {path} ({len(state)}개 텐서)")


def load_lora_checkpoint(model: nn.Module, path: str, strict: bool = False):
    """저장된 LoRA 체크포인트를 모델에 로드."""
    state = torch.load(path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=strict)
    print(f"[LoRA] 체크포인트 로드: {path}")
    if missing:
        print(f"  missing keys : {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  unexpected   : {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
