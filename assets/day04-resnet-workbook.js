window.resnetWorkbook = [
  {
    "step": "step-1",
    "title": "1단계 코드 연습 · 이미지와 라벨은 한 쌍",
    "source": "시험 범위 내 · 원본 절 1) Dataset | 모의시험 VISION02의 데이터 구성 부분 · 원본 노트북 셀 6 · 작은 텐서로 바꾼 예제",
    "theory": [
      "이미지는 모델이 읽는 입력이고, 라벨은 맞혀야 할 정답 번호입니다. RGB의 3과 클래스의 10은 서로 다른 뜻입니다.",
      "예시: 고양이와 강아지 두 장이면 images는 (2,3,32,32), labels는 [3,5]입니다. 라벨을 이미지 채널에 붙이지 않고 Dataset에서 짝으로 묶습니다."
    ],
    "shape": "images (N,3,32,32) float32 + labels (N,) long → Dataset 샘플: (3,32,32)와 scalar 라벨",
    "question": "RGB 채널 수와 클래스 수는 왜 다른가요? 이미지 한 장에 붙는 라벨의 개수도 설명하세요.",
    "task": "필요한 import는 아래 코드 맨 위에 표시했습니다. 제공된 images·labels를 짝으로 묶으세요. 이 문제의 images는 이미 텐서로 변환된 입력입니다.",
    "code": "import torch\n\ndef pair_images_labels(images, labels):\n    labels = labels.to(dtype=torch.????[1])\n    dataset = torch.utils.data.TensorDataset(????[2], ????[3])\n    return dataset",
    "blanks": [
      {
        "label": "라벨의 정수 dtype",
        "answers": [
          "long",
          "int64"
        ]
      },
      {
        "label": "첫 번째 데이터",
        "answers": [
          "images"
        ]
      },
      {
        "label": "두 번째 데이터",
        "answers": [
          "labels"
        ]
      }
    ],
    "hint": "이미지는 실수 픽셀, 이 분류 문제의 정답은 정수 클래스 번호입니다. Dataset은 같은 index의 입력과 정답을 반환합니다.",
    "explanation": "TensorDataset(images, labels)는 index i로 images[i], labels[i]를 함께 꺼냅니다. 라벨 dtype은 정수 인덱스를 받는 CrossEntropyLoss와 연결됩니다.",
    "theoryAnswer": "3은 RGB 색상 성분 개수, 10은 분류할 종류의 수입니다. 단일 라벨 분류에서는 이미지마다 정답 번호 하나가 있습니다.",
    "transfer": "Vision 문제지 VISION02에서 labels dtype과 Dataset의 반환값을 확인합니다."
  },
  {
    "step": "step-2",
    "title": "2단계 코드 연습 · train과 test의 변환 구분",
    "source": "시험 범위 내 · 원본 절 1) Dataset | 모의시험 VISION01을 강사 강조 코드 형태로 재구성 · 원본 셀 6",
    "theory": [
      "T는 torchvision.transforms의 별명, tfms는 transforms의 줄임말입니다. Compose는 각 변환을 위에서 아래 순서대로 적용합니다.",
      "학습용 증강은 같은 이미지의 위치·방향을 조금 바꿔 보여줍니다. 평가는 랜덤 증강을 빼서 입력 처리를 일관되게 합니다. 둘 다 같은 mean/std로 정규화합니다.",
      "예시: padding=4는 사방에 4칸을 채워 32→40으로 만들고, RandomCrop(32)는 그 안에서 32×32를 뽑습니다. 기본 좌우 반전 확률은 0.5입니다. Normalize는 (값-mean)/std로 값을 바꾸며 모양은 유지합니다."
    ],
    "shape": "PIL RGB → (3,32,32) float32. Compose만으로 배치 축은 생기지 않습니다.",
    "question": "test에서 빼야 하는 두 변환과 그 이유를 말하세요. ToTensor와 Normalize의 차이는 무엇인가요?",
    "task": "필요한 import는 아래 코드 맨 위에 표시했습니다. mean/std는 RGB 3채널 값입니다. train에만 두 랜덤 증강을 넣고 두 파이프라인을 반환하세요.",
    "code": "import torchvision\n\ndef make_transforms(mean, std):\n    train_tfms = torchvision.transforms.Compose([\n        torchvision.transforms.????[1](32, padding=4),\n        torchvision.transforms.????[2](),\n        torchvision.transforms.????[3](),\n        torchvision.transforms.Normalize(mean, std),\n    ])\n    test_tfms = torchvision.transforms.Compose([\n        torchvision.transforms.????[4](),\n        torchvision.transforms.????[5](mean, std),\n    ])\n    return train_tfms, test_tfms",
    "blanks": [
      {
        "label": "랜덤 위치에서 자르기",
        "answers": [
          "RandomCrop"
        ]
      },
      {
        "label": "좌우 반전",
        "answers": [
          "RandomHorizontalFlip"
        ]
      },
      {
        "label": "학습 이미지 텐서 변환",
        "answers": [
          "ToTensor"
        ]
      },
      {
        "label": "평가 이미지 텐서 변환",
        "answers": [
          "ToTensor"
        ]
      },
      {
        "label": "평가 이미지 정규화",
        "answers": [
          "Normalize"
        ]
      }
    ],
    "hint": "랜덤 변환은 train에만 있습니다. Normalize가 텐서를 받도록 ToTensor 뒤에 배치합니다.",
    "explanation": "test에도 Normalize가 들어갑니다. 이 uint8 PIL 입력에서 ToTensor는 CHW 실수 텐서와 0~1 범위를 만들고 Normalize는 채널별 값을 다시 조정합니다. 정규화 결과는 0~1로 제한되지 않습니다.",
    "theoryAnswer": "Crop·Flip은 학습 데이터 증강입니다. 평가에서는 빼지만 ToTensor·Normalize는 유지합니다. mean/std를 평가 이미지에서 다시 계산하지 않습니다.",
    "transfer": "VISION01을 풀고 직사각형 평가 이미지에서도 H,W 순서를 확인합니다."
  },
  {
    "step": "step-3",
    "title": "3단계 코드 연습 · DataLoader의 마지막 배치",
    "source": "시험 범위 내 · 원본 절 1) Dataset | 모의시험 VISION02 · 원본 셀 6",
    "theory": [
      "Dataset은 한 샘플을, DataLoader는 여러 샘플을 묶은 배치를 제공합니다. shuffle은 꺼내는 순서를 바꾸며 이미지와 정답의 짝은 함께 유지됩니다.",
      "예시: 데이터 5장을 batch_size=2, drop_last=False로 읽으면 2장·2장·1장입니다. 모델 코드는 마지막 배치도 처리하도록 B를 고정하지 않습니다."
    ],
    "shape": "샘플 (3,32,32), scalar → 배치 (B,3,32,32), labels (B,)",
    "question": "이미지 5장, batch_size=2에서 마지막 images·labels shape를 적으세요. drop_last=True로 바꾸면 어떻게 되나요?",
    "task": "필요한 import는 아래 코드 맨 위에 표시했습니다. images와 labels는 제공됩니다. 학습일 때만 순서를 섞고 마지막 작은 배치는 남기세요.",
    "code": "import torch\n\ndef vision_loader(images, labels, batch_size, train):\n    dataset = torch.utils.data.TensorDataset(images, labels)\n    return torch.utils.data.DataLoader(\n        ????[1], batch_size=????[2],\n        shuffle=????[3], drop_last=????[4], num_workers=0\n    )",
    "blanks": [
      {
        "label": "묶을 Dataset",
        "answers": [
          "dataset"
        ]
      },
      {
        "label": "배치 크기",
        "answers": [
          "batch_size"
        ]
      },
      {
        "label": "학습 여부에 따른 shuffle",
        "answers": [
          "train"
        ]
      },
      {
        "label": "마지막 작은 배치도 남기기",
        "answers": [
          "False"
        ]
      }
    ],
    "hint": "인자로 받은 값을 사용하세요. drop_last는 마지막 불완전 배치를 버릴지 정하는 bool입니다.",
    "explanation": "train=True면 섞고, False면 원래 순서로 읽습니다. num_workers=0은 현재 프로세스에서 데이터를 읽는 설정입니다.",
    "theoryAnswer": "마지막 images=(1,3,32,32), labels=(1,)입니다. drop_last=True면 마지막 1장을 버려 두 배치만 남습니다.",
    "transfer": "VISION02의 확인 셀에서 [2,2,1] 배치 크기와 라벨 순서를 검사합니다."
  },
  {
    "step": "step-4",
    "title": "4단계 코드 연습 · CIFAR용 첫 Conv 만들기",
    "source": "시험 범위 내 · 원본 절 2) 모델 준비 | 모의시험 VISION03의 stem 부분 · 원본 셀 9",
    "theory": [
      "Conv2d의 첫 두 인자는 입력 채널과 출력 채널입니다. 필터는 입력 채널 전체에 걸쳐 계산하므로 RGB 입력의 첫 필터 한 개는 (3,3,3) 가중치를 가집니다.",
      "예시: Conv2d(3,64,3,stride=1,padding=1)의 가중치는 (64,3,3,3), 출력은 (B,64,32,32)입니다. 출력 채널 64는 사진 수가 아니라 이미지당 특징판 수입니다."
    ],
    "shape": "(B,3,32,32) → (B,64,32,32)",
    "question": "64는 무엇의 개수인가요? padding=0이라면 32×32 입력의 출력 공간 크기는 얼마인가요?",
    "task": "필요한 import는 아래 코드 맨 위에 표시했습니다. 출력 크기를 보존하는 첫 3×3 Conv로 바꾸고 기존 maxpool을 제거하세요. FC는 뒤 단계에서 다룹니다.",
    "code": "import torch\nimport torchvision\n\ndef build_cifar_stem():\n    model = torchvision.models.resnet18(weights=None)\n    model.conv1 = torch.nn.Conv2d(\n        ????[1], ????[2], kernel_size=3,\n        stride=????[3], padding=????[4], bias=False\n    )\n    model.maxpool = torch.nn.????[5]()\n    return model",
    "blanks": [
      {
        "label": "입력 채널",
        "answers": [
          "3"
        ]
      },
      {
        "label": "출력 채널",
        "answers": [
          "64"
        ]
      },
      {
        "label": "이동 간격",
        "answers": [
          "1"
        ]
      },
      {
        "label": "테두리 채우기",
        "answers": [
          "1"
        ]
      },
      {
        "label": "그대로 통과시키는 모듈",
        "answers": [
          "Identity"
        ]
      }
    ],
    "hint": "dilation=1일 때 out=floor((in+2p-k)/s)+1입니다. maxpool 자리는 항등 모듈로 바꿉니다.",
    "explanation": "작은 CIFAR 입력의 공간 해상도가 너무 빨리 줄어들지 않게 stride=1과 Identity를 사용합니다. weights=None은 사전학습 가중치를 불러오지 않는 설정입니다.",
    "theoryAnswer": "64개 필터가 특징판 64개를 만듭니다. padding=0이면 (32-3)+1=30이라 출력은 (B,64,30,30)입니다.",
    "transfer": "VISION03에서 같은 stem과 최종 FC를 함께 구성합니다."
  },
  {
    "step": "step-5",
    "title": "5단계 코드 연습 · residual 덧셈 완성",
    "source": "시험 범위 내 · 원본 절 2) 모델 준비 | VISION03의 구조 이해를 위한 보충 부분 구현 · torchvision BasicBlock·교안 37~40쪽",
    "theory": [
      "우회 경로는 입력 정보를 본 경로의 변환 결과에 더합니다. concat이 아닌 원소별 덧셈이므로 두 경로의 shape를 맞춰야 합니다.",
      "예시: 입력 (B,64,32,32)을 본 경로가 (B,128,16,16)으로 바꾸면 우회 경로도 projection을 통해 같은 shape로 바뀌어야 합니다. 덧셈 뒤 채널은 128입니다."
    ],
    "shape": "본문과 우회 경로의 동일한 (B,C,H,W) → 덧셈·ReLU 후 같은 shape",
    "question": "x와 F(x)를 더하는 것과 채널 축으로 붙이는 것은 무엇이 다른가요? downsample은 언제 필요한가요?",
    "task": "필요한 import는 아래 코드 맨 위에 표시했습니다. 아래는 BasicBlock의 forward 끝부분을 떼어 낸 보충 함수입니다. residual에는 본 경로의 Conv·BN 계산이 이미 끝났습니다. downsample은 None 또는 입력을 같은 shape로 바꾸는 모듈입니다.",
    "code": "import torch\n\ndef residual_finish(x, residual, downsample=None):\n    identity = x\n    if downsample is not None:\n        identity = ????[1](x)\n    out = residual ????[2] identity\n    return torch.????[3](out)",
    "blanks": [
      {
        "label": "우회 경로 shape 변환",
        "answers": [
          "downsample"
        ]
      },
      {
        "label": "두 경로를 합치는 연산자",
        "answers": [
          "+"
        ]
      },
      {
        "label": "마지막 활성화 함수",
        "answers": [
          "relu"
        ]
      }
    ],
    "hint": "필요할 때 우회 경로도 변환합니다. 덧셈 후 음수는 0으로 바꿉니다.",
    "explanation": "downsample은 여기서 공간 축소뿐 아니라 채널 수 변경도 포함할 수 있습니다. 실제 layer2 첫 블록은 1×1 Conv(stride=2)와 BN을 씁니다.",
    "theoryAnswer": "덧셈은 같은 칸끼리 더해 shape를 유지하고, 채널 concat은 채널 수를 합칩니다. 본 경로의 채널·공간 크기가 바뀌는 경우 우회 경로도 맞춥니다.",
    "transfer": "VISION03의 shape 결과를 보고 layer2 첫 블록의 downsample을 출력해 확인합니다."
  },
  {
    "step": "step-6",
    "title": "6단계 코드 연습 · stage 순서 연결",
    "source": "시험 범위 내 · 원본 절 2) 모델 준비 | VISION03의 특징 추출 흐름을 분리한 보충 함수 · 노트북 셀 9의 ResNet 구조",
    "theory": [
      "layer1~4는 블록의 묶음입니다. ResNet18에서는 각각 기본 블록 2개가 있으며, layer2~4의 첫 블록이 채널·공간 크기를 바꿉니다.",
      "예시: layer2가 128×16×16을 만들면 layer3는 256×8×8, layer4는 512×4×4를 만듭니다. 모든 블록에서 공간 크기를 줄이는 것은 아닙니다."
    ],
    "shape": "stem (B,64,32,32) → layer1 같은 shape → layer2 (B,128,16,16) → layer3 (B,256,8,8) → layer4 (B,512,4,4)",
    "question": "layer2의 첫 블록과 두 번째 블록은 공간 크기에 어떤 차이를 주나요? 배치 수 B도 절반이 되나요?",
    "task": "필요한 import는 아래 코드 맨 위에 표시했습니다. model은 CIFAR용 ResNet18, x는 stem을 이미 통과한 특징입니다. 네 stage를 순서대로 적용하세요.",
    "code": "# 추가 import 없음: model과 x는 함수 인자로 받습니다.\n\ndef run_stages(model, x):\n    x = model.layer1(x)\n    x = model.????[1](x)\n    x = model.????[2](x)\n    x = model.????[3](x)\n    return x",
    "blanks": [
      {
        "label": "두 번째 stage",
        "answers": [
          "layer2"
        ]
      },
      {
        "label": "세 번째 stage",
        "answers": [
          "layer3"
        ]
      },
      {
        "label": "네 번째 stage",
        "answers": [
          "layer4"
        ]
      }
    ],
    "hint": "단계 이름과 출력 채널 64→128→256→512를 연결하세요.",
    "explanation": "model.layer2는 Conv 하나가 아니라 두 블록을 포함하는 모듈입니다. 앞 stage의 출력을 다음 stage의 입력으로 넘깁니다.",
    "theoryAnswer": "첫 블록에서 stride=2로 공간을 절반으로 바꾸고 두 번째 블록은 유지합니다. B는 이미지 개수이므로 바뀌지 않습니다.",
    "transfer": "VISION03의 forward hook 결과가 위 stage 표와 맞는지 확인합니다."
  },
  {
    "step": "step-7",
    "title": "7단계 코드 연습 · 평균과 flatten 순서",
    "source": "시험 범위 내 · 원본 절 2) 모델 준비 | 기존 resnet-shape-check.md의 pool_and_flatten 과제 · torchvision ResNet forward",
    "theory": [
      "AdaptiveAvgPool2d((1,1))은 채널마다 공간 값을 평균내어 한 숫자로 줄입니다. 채널끼리 평균내거나 배치끼리 섞지 않습니다.",
      "예시: 한 채널의 [[1,3],[5,7]]은 평균 4입니다. 채널이 512개라면 이미지당 평균값도 512개입니다. flatten(start_dim=1)은 배치 축을 보존합니다."
    ],
    "shape": "(B,C,H,W) → (B,C,1,1) → (B,C)",
    "question": "평균 전에 바로 flatten하면 이 모델의 FC에 왜 넣을 수 없나요? start_dim=0이면 어떤 문제가 생기나요?",
    "task": "필요한 import는 아래 코드 맨 위에 표시했습니다. 특징 채널 C와 공간 크기를 하드코딩하지 않고 출력 공간 1×1로 평균낸 뒤 배치 축을 남기세요.",
    "code": "import torch\n\ndef pool_and_flatten(features):\n    pool = torch.nn.AdaptiveAvgPool2d(????[1])\n    pooled = ????[2](features)\n    return torch.flatten(pooled, start_dim=????[3])",
    "blanks": [
      {
        "label": "출력 공간 크기",
        "answers": [
          "(1, 1)",
          "1",
          "[1, 1]"
        ]
      },
      {
        "label": "평균 모듈 적용",
        "answers": [
          "pool"
        ]
      },
      {
        "label": "펼치기 시작 축",
        "answers": [
          "1"
        ]
      }
    ],
    "hint": "B축은 0번입니다. 마지막의 C×1×1만 펼쳐야 합니다.",
    "explanation": "공간 크기를 먼저 줄이고 펼치므로 (B,512,4,4)는 (B,512)가 됩니다. F.adaptive_avg_pool2d를 이용한 동등한 풀이도 Jupyter에서 가능합니다.",
    "theoryAnswer": "먼저 flatten하면 이미지당 8192개라 512개를 기대하는 FC와 맞지 않습니다. start_dim=0이면 배치까지 한 줄로 합쳐 이미지별 구분이 사라집니다.",
    "transfer": "pool_and_flatten을 빈 함수 본문으로 다시 작성하고 입력 C,H,W를 바꿔도 동작하는지 봅니다."
  },
  {
    "step": "step-8",
    "title": "8단계 코드 연습 · 클래스 수에 맞는 FC",
    "source": "시험 범위 내 · 원본 절 2) 모델 준비 | 모의시험 VISION03의 head 부분 · 원본 셀 9",
    "theory": [
      "Linear는 마지막 특징 축을 클래스별 점수 축으로 바꿉니다. weight의 저장 shape는 (출력 특징 수, 입력 특징 수)입니다.",
      "예시: (B,512) 입력을 10클래스로 분류하면 Linear(512,10), weight=(10,512), logits=(B,10)입니다. weight는 배치마다 새로 만들지 않고 여러 이미지에 공통 적용합니다."
    ],
    "shape": "features (B,512) → logits (B,num_classes)",
    "question": "클래스를 10개에서 3개로 바꾸면 FC weight와 logits shape는 어떻게 바뀌나요?",
    "task": "필요한 import는 아래 코드 맨 위에 표시했습니다. 기존 ResNet18의 fc를 새 클래스 수에 맞게 교체하세요. num_classes를 10으로 고정하지 마세요.",
    "code": "import torch\n\ndef replace_head(model, num_classes):\n    in_dim = model.fc.????[1]\n    model.fc = torch.nn.????[2](in_dim, ????[3])\n    return model",
    "blanks": [
      {
        "label": "기존 FC의 입력 특징 개수",
        "answers": [
          "in_features"
        ]
      },
      {
        "label": "완전 연결 레이어",
        "answers": [
          "Linear"
        ]
      },
      {
        "label": "반환할 클래스 점수 개수",
        "answers": [
          "num_classes"
        ]
      }
    ],
    "hint": "이미지 특징 개수는 기존 fc에서 읽고, 출력 개수는 함수 인자를 사용합니다.",
    "explanation": "이 ResNet의 in_features는 512입니다. 새 Linear의 파라미터는 새로 초기화되며, 출력 점수는 아직 확률이 아닙니다.",
    "theoryAnswer": "weight=(3,512), bias=(3,), logits=(B,3)입니다. B와 입력 특징 수 512는 유지됩니다.",
    "transfer": "VISION03의 build_cifar_resnet을 10클래스와 3클래스 두 인자로 검사합니다."
  },
  {
    "step": "step-9",
    "title": "9단계 코드 연습 A · 학습과 평가 연결",
    "source": "시험 범위 내 · 원본 절 3~4) 학습 유틸리티·학습 | 모의시험 VISION04 · 원본 셀 11 · CPU 핵심 흐름으로 AMP 생략",
    "theory": [
      "train/eval은 BatchNorm·Dropout 등의 동작 모드입니다. eval만 호출해도 gradient 기록이 자동으로 꺼지는 것은 아니므로 평가에서는 gradient 기록도 끕니다.",
      "학습은 이전 gradient 초기화→순전파·loss→backward→optimizer.step 순서입니다. backward는 기울기를 계산하고 step은 그 기울기로 파라미터를 바꿉니다.",
      "예시: 배치가 2장과 1장일 때 배치 평균 loss가 1과 4라면 전체 샘플 평균은 (1×2+4×1)/3=2입니다. 두 평균을 그대로 더해 2로 나누면 2.5로 틀립니다."
    ],
    "shape": "images (B,3,32,32), labels (B,) → logits (B,10) → loss scalar, pred (B,)",
    "question": "eval과 no_grad의 역할은 어떻게 다른가요? CrossEntropyLoss 입력에 pred 대신 logits를 넣는 이유도 적으세요.",
    "task": "필요한 import는 아래 코드 맨 위에 표시했습니다. 모의시험 함수입니다. criterion은 배치 평균 CrossEntropyLoss이고 loader는 비어 있지 않습니다. optimizer가 있으면 학습, None이면 평가합니다. 코드 흐름을 먼저 설명한 뒤 빈칸을 채우세요.",
    "code": "import torch\n\ndef classification_epoch(model, loader, criterion, device, optimizer=None):\n    training = optimizer is not None\n    model.train(????[1])\n    total_loss, correct, count = 0.0, 0, 0\n    with torch.set_grad_enabled(????[2]):\n        for images, labels in loader:\n            images, labels = images.to(device), labels.to(device)\n            if training:\n                optimizer.????[3]()\n            logits = model(images)\n            loss = criterion(????[4], labels)\n            if training:\n                loss.????[5]()\n                optimizer.????[6]()\n            bs = labels.size(0)\n            total_loss += loss.item() * ????[7]\n            correct += (logits.argmax(dim=????[8]) == labels).sum().item()\n            count += bs\n    return total_loss / count, correct / count",
    "blanks": [
      {
        "label": "모델 동작 모드",
        "answers": [
          "training"
        ]
      },
      {
        "label": "gradient 기록 여부",
        "answers": [
          "training"
        ]
      },
      {
        "label": "이전 gradient 초기화",
        "answers": [
          "zero_grad"
        ]
      },
      {
        "label": "손실함수 입력 점수",
        "answers": [
          "logits"
        ]
      },
      {
        "label": "역전파",
        "answers": [
          "backward"
        ]
      },
      {
        "label": "파라미터 갱신",
        "answers": [
          "step"
        ]
      },
      {
        "label": "배치 평균을 샘플 합으로 변환",
        "answers": [
          "bs",
          "labels.size(0)",
          "labels.shape[0]"
        ]
      },
      {
        "label": "클래스 축",
        "answers": [
          "1",
          "-1"
        ]
      }
    ],
    "hint": "두 training은 모드와 autograd를 각각 제어합니다. loss에는 raw logits를 넣습니다. 확률 표시가 필요할 때만 별도 softmax를 계산합니다.",
    "explanation": "평가에서는 optimizer 작업을 생략하고 gradient 기록을 끕니다. 학습에서는 gradient를 매 배치 초기화합니다. loss와 정답 수는 실제 샘플 수로 나눠 마지막 작은 배치도 올바르게 집계합니다.",
    "theoryAnswer": "eval은 모듈의 학습/평가 동작, no_grad는 연산 그래프 기록을 제어합니다. pred는 선택된 번호뿐이고 logits는 모든 클래스 점수라 정답과의 오차를 학습에 연결할 수 있습니다.",
    "transfer": "VISION04의 공개 확인 셀을 실행해 학습 시 weight 변경과 평가 시 보존을 확인합니다."
  },
  {
    "step": "step-9",
    "title": "9단계 코드 연습 B · 어디에서 틀렸는지 집계",
    "source": "시험 범위 내 · 원본 절 6) 성능 분석 | 모의시험 VISION05 · 원본 셀 17 · 이론과 코딩을 연결하는 추가 문제",
    "theory": [
      "전체 accuracy는 맞힌 이미지 수/전체 이미지 수입니다. 혼동행렬은 어떤 실제 클래스를 어떤 클래스로 잘못 골랐는지 보여줍니다.",
      "예시: 실제 고양이 3번을 강아지 5번으로 예측했으면 cm[3,5]에 1을 더합니다. 행은 실제 라벨, 열은 예측입니다. 대각선은 맞힌 경우입니다."
    ],
    "shape": "preds,labels (N,) → cm (C,C), 클래스별 정답 비율 (C,), 전체 accuracy float",
    "question": "클래스별 정답 비율을 계산할 때 왜 열 합이 아니라 행 합으로 나누나요?",
    "task": "필요한 import는 아래 코드 맨 위에 표시했습니다. preds와 labels는 CPU 정수 텐서이고 N은 1 이상입니다. 실제 샘플이 없는 클래스의 비율은 0으로 둡니다.",
    "code": "import torch\n\ndef classification_metrics(preds, labels, num_classes):\n    cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)\n    for truth, pred in zip(labels, preds):\n        cm[int(????[1]), int(????[2])] += 1\n    per_class = cm.diagonal().float() / cm.sum(dim=????[3]).clamp_min(1)\n    accuracy = (preds == labels).float().????[4]().item()\n    return cm, per_class, accuracy",
    "blanks": [
      {
        "label": "행 번호",
        "answers": [
          "truth"
        ]
      },
      {
        "label": "열 번호",
        "answers": [
          "pred"
        ]
      },
      {
        "label": "행마다 합을 구할 축",
        "answers": [
          "1",
          "-1"
        ]
      },
      {
        "label": "전체 정답 비율 연산",
        "answers": [
          "mean"
        ]
      }
    ],
    "hint": "행 하나는 실제 특정 클래스인 샘플들을 모은 것입니다. True는 float 변환 후 1, False는 0입니다.",
    "explanation": "cm.sum(dim=1)은 각 실제 클래스의 샘플 수입니다. 대각선 값은 그중 맞힌 개수입니다. 분모를 최소 1로 두어 샘플이 없는 행에서 0/0을 피합니다.",
    "theoryAnswer": "분모는 해당 실제 클래스의 전체 샘플 수여야 하므로 행 합입니다. 열 합은 그 클래스로 예측한 샘플 수여서 다른 지표가 됩니다.",
    "transfer": "VISION05에서 confusion matrix의 행·열 방향과 없는 클래스 처리를 확인합니다."
  }
];
