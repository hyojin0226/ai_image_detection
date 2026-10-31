"""Phase 3 비교 모델 (PyTorch): EfficientNetV2S / ConvNeXt-T / CLIP 선형 분류기."""

import torch
from torch import nn

# timm 모델 이름 (ImageNet 사전학습)
CNN_MODELS = {
    "efficientnetv2s": "tf_efficientnetv2_s.in21k_ft_in1k",
    "convnext_tiny": "convnext_tiny.fb_in22k_ft_in1k",
}

CLIP_MODEL = ("ViT-L-14", "openai")


def build_cnn(name: str) -> nn.Module:
    """출력 1개(logit, real일 확률의 로짓)인 분류 모델."""
    import timm

    return timm.create_model(CNN_MODELS[name], pretrained=True, num_classes=1)


def cnn_transform(name: str, train: bool):
    """timm 모델별 정규화. 입력 이미지는 data.load_image()에서 이미 256x256으로 맞춰져 있습니다."""
    import timm
    from torchvision import transforms

    config = timm.data.resolve_data_config({}, model=timm.create_model(CNN_MODELS[name], pretrained=False))
    steps = []
    if train:
        # 1차 실험과 같은 약한 증강
        steps += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=7, translate=(0.03, 0.03), scale=(0.95, 1.05)),
            transforms.ColorJitter(contrast=0.10),
        ]
    steps += [transforms.ToTensor(), transforms.Normalize(config["mean"], config["std"])]
    return transforms.Compose(steps)


def build_clip(device: str = "cuda"):
    """백본 고정 CLIP 이미지 인코더와 전처리 함수를 반환합니다."""
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(CLIP_MODEL[0], pretrained=CLIP_MODEL[1])
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model, preprocess


@torch.no_grad()
def clip_features(model, images: torch.Tensor) -> torch.Tensor:
    """L2 정규화된 이미지 특징 (UnivFD와 같은 방식)."""
    features = model.encode_image(images)
    return features / features.norm(dim=-1, keepdim=True)
