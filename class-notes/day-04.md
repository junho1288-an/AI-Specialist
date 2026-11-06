# Day 04 · Vision 이어서 · Swin Transformer 이론 시작

- 날짜: 2026-11-05 (Asia/Seoul)
- 오늘 분야: Vision 이어서 진행.
- 이론 시작: [03_Attention.pdf](../vision/03_Attention.pdf) **30쪽부터**. PDF 파일의 30번째 페이지는 Swin Transformer의 도입 배경이다.
- 현재 이론 진도(후속 확인): [04_Applications.pdf](../vision/04_Applications.pdf) 8쪽 부근, 객체 탐지 평가 지표. 설명 근거는 6~9쪽이다.
- 실습 진도 최신 확인: 사용자 전달 기준 **U-Net까지 진행 완료**. [04_Unet.ipynb](../vision/04_Unet.ipynb)의 패딩 사용을 이후 설명·문제에 반영한다. 수업 실습 진행과 자료 없이 독립 구현 가능한지는 구분한다.
- 직접 작성할 실습 위치: [practice/day-04](../practice/day-04/README.md)
- 이전 필기: [Day 3 · Vision](day-03.md)

## 오늘 진행 방식

- 오늘부터 질문·짧은 설명·예시는 이 파일에 기록한다. 새 실습과 답안은 `practice/day-04/`에 저장한다.
- 설명은 핵심을 짧게 정리하고 예시 1개를 포함한다. 코드에서는 입력·출력 shape, dtype, 연산 이유를 함께 확인한다.
- 시험 연습은 `????` 빈칸·코드 줄 완성·제공된 클래스나 함수의 일부 구현으로 진행한다. [시험 범위](../EXAM-SCOPE.md)를 적용한다.
- 기록 예시: `교안 파일 / 12쪽 / 노트북 셀 5 / 입력 shape가 바뀌는 이유 질문`.

## 시작 전 짧은 복습 · 3~5분

정답을 먼저 보지 않고 내 말로 답한다. 현재 둘 다 미응답이다.

1. **전날 Vision:** 32×32 RGB 이미지 8장을 PyTorch 배치로 묶으면 입력 shape는? 10클래스 분류 모델의 logits와 정답 labels shape는 각각?
   - 내 답:
   - 도움·피드백:
2. **3수업일 전 Day 1:** 토큰 ID가 `[10,20,30,40,50]`일 때 context_size=4인 첫 입력 x와 다음 토큰 정답 y는? y를 한 칸 옮기는 이유는?
   - 내 답:
   - 도움·피드백:

7수업일 전 복습은 아직 해당하지 않는다. 오늘 이론은 Swin Transformer 도입부터 시작하며, 작은 학습 목표와 과제 난도는 질문·위 답변을 확인한 뒤 맞춘다.

## 수업 중 메모

- 교안 위치: `vision/03_Attention.pdf` 30쪽부터. 오늘 종료 페이지는 수업 후 기록.
- 강사 설명:
- 내 질문:

### 시작 페이지 확인 · 교안 30쪽

- **Swin Transformer**의 Swin은 **Shifted Window(이동한 윈도우)**에서 온 이름이다. 교안은 전체 패치 사이 attention의 계산량과 계층적 이미지 표현 문제를 도입 배경으로 설명한다.
- **보충 예시:** 이미지 패치 전체를 한 번에 비교하는 대신 작은 구역 안에서 먼저 비교하고, 다음 블록에서 구역 경계를 옮겨 이웃 구역과 정보가 섞이게 한다. 윈도우 이동 설명은 다음 31쪽과 연결된다.
- 에이전트가 30~32쪽 본문을 미리 확인한 것이며, 사용자가 32쪽까지 수강·이해했다는 기록은 아니다.

## 오늘 핵심 3줄

1.
2.
3.

## 코드 흐름과 이유

| 단계·함수 | 내 말로 설명 | 입력 → 출력 shape |
| --- | --- | --- |
| | | |

## 강사의 시험 관련 언급

- **ResNet 실습은 6번 성능 분석까지 포함(사용자 전달).** 노트북 전체 셀 17까지이며 이후 Activation Map·커널 시각화·Grad-CAM·Receptive Field·t-SNE·ONNX는 제외한다. HTML의 9단계는 설명 순서라 노트북 절 번호와 다르다. **예시:** HTML 8단계 FC는 노트북 2번 모델 준비에 해당하므로 시험 대비에 포함한다. HTML 상단에 대응표와 범위를 추가하고 각 코드 문제에 해당 절을 표시했다.
- **DETR 실습 범위 확정(사용자 전달): 13번까지 포함.** 노트북의 `13) Encoder self-attention 시각화`까지이며, 현재 전체 셀 번호로는 30번 코드 셀까지다. `14) AttentionVisualizer 클래스`와 `15) 인터랙티브 시각화 실행`은 제외한다. **예시:** 기존 모의시험 VISION12의 confidence 필터·박스 좌표 복원은 13번 이전 내용이라 유지한다. PDF의 매칭·loss 이론은 이번 실습 제한만으로 제외하지 않는다.
- **실습 중 강조(사용자 전달):** `train_tfms`와 `test_tfms`의 Compose 구성 차이를 잘 보라고 했다. 개별 문항 출제 확정은 아니다. 아래 전처리 비교를 복습 후보로 기록한다.
- **사용자 요청으로 CLIP 코드 시험 대비 후보 추가:** 현재 PDF의 코드 위치는 42쪽 Training (1/2)이다. 사용자가 지칭한 32쪽은 현재 파일에서 Swin 구조여서 페이지 차이를 함께 기록한다. 개별 코드의 출제 확정 안내는 아니다.
- [CLIP 코드·shape·시험 후보 정리](../vision/day04-clip-exam-candidate.md): 이미지·문장 특징 → 공통 공간 투영·정규화 → `(N,N)` 유사도 → 양방향 CE 평균. **예시:** 고양이·강아지 이미지와 설명을 같은 순서로 두면 정답 위치는 `[0,1]`이다.
- 실제 안내·교안 위치:
- 아직 확인하지 않은 예상:

## 질문과 오류

### ResNet 실습 0~8번 줄별 해설 노트북

- [새 Jupyter 해설](../practice/day-04/resnet18-line-by-line.ipynb): 원본 코드의 각 줄을 읽고, 이미지 흐름·F(x)+x·loss에서 가중치 갱신까지 연결한다. 예: `scaler.step(optimizer)`는 가중치 갱신, `scaler.update()`는 확대 배율 갱신이다.
- 환경 준비부터 전처리·모델·학습·추론·혼동행렬·Activation Map·Kernel까지 설명한다. **시험 범위는 여전히 6번까지**이며 7~8번은 추가 학습이다.
- 원본 코드는 읽기용 Markdown, 실행 셀 9개는 다운로드 없는 CPU 합성 예제다. 새 커널 전체 실행과 shape·기울기·혼동행렬 확인을 통과했고 출력·그림을 저장했다. CIFAR 전체 학습·CUDA AMP 실행을 검증했다는 뜻은 아니다. Day 4 요청을 2026-11-06에 이어 완료한 자료다.

### `train_one_epoch`: 순전파·loss·역전파·가중치 갱신

- **전체 흐름:** images `(B,3,32,32)` → 첫 Conv·BN·ReLU → 잔차 블록들 `ReLU(F(x)+x)` → `(B,512,4,4)` → 평균 풀링·flatten `(B,512)` → FC → logits `(B,10)` → labels `(B,)`와 criterion으로 비교 → scalar loss → backward로 gradient 계산 → optimizer로 가중치 수정. `model(images)`가 모델의 forward 전체를 실행한다. `criterion`은 손실함수 객체를 담은 변수 이름이며 이 실습은 `torch.nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)`다. logits는 확률·예측 번호가 아닌 원본 점수이며 CE 앞에 softmax를 따로 넣지 않는다.
- **optimizer와 zero_grad:** optimizer는 gradient와 학습률 등으로 파라미터를 갱신하는 객체이며 실습은 SGD다. `zero_grad(set_to_none=True)`는 이전 gradient를 None으로 비운다. 가중치 자체를 0으로 만들지 않는다. backward는 기본적으로 gradient를 누적하므로 배치별 갱신에서는 비워야 한다. 보충 예시: 이전 gradient=2, 이번=3이면 비우지 않을 때 5가 된다. 단순 SGD 예시 `w=2, grad=.3, lr=.1`이면 `w=2-.1*.3=1.97`이며 실제 실습에는 momentum·weight decay도 있다.
- **AMP와 scaler:** autocast는 연산별로 정밀도를 선택하는 자동 혼합 정밀도(Automatic Mixed Precision) 문맥이다. GradScaler는 작은 FP16 gradient가 사라지는 문제를 줄이도록 loss를 확대한다. `scaler.scale(loss).backward()`는 확대된 gradient 계산, `scaler.step(optimizer)`는 gradient 배율을 되돌리고 정상일 때 optimizer.step 호출(비정상 inf/NaN이면 건너뜀), `scaler.update()`는 다음 배치의 확대 배율 갱신이다. **가중치 갱신은 step이며 update는 배율 갱신**이다. 예: 배율 1024면 역전파 gradient도 1024배, optimizer 적용 전에 다시 나눈다. 원본 loss 값 자체를 덮어쓰지 않는다.
- **준비·집계 줄:** model.train은 BN·Dropout의 학습 모드 설정이며 자체로 가중치를 학습하지 않는다. 누적 변수와 n을 0으로 시작하고 time.time으로 시간을 잰다. loader에서 배치를 꺼내 images·labels를 device로 이동한다. non_blocking=True는 가능한 경우 비동기 전송을 요청하며 항상 연산과 겹친다고 보장하지 않는다. images.size(0)은 실제 배치 크기, loss.item은 scalar 텐서를 Python 숫자로 꺼내기다. 평균 loss에 bs를 곱해 샘플 합으로 누적한다. accuracy_top1은 0~1 정답 비율이므로 bs를 곱하면 정답 수다. logits.detach는 gradient 연결을 끊되 저장 공간을 공유하므로 복사라는 주석은 부정확하다. 예: 4장 평균 loss=2, 1장 평균 loss=1이면 전체 평균은 `(2*4+1*1)/5=1.8`이다. n은 처리 샘플 수, dt는 경과 시간이며 마지막에 평균 loss·정확도·시간을 반환한다. 빈 loader는 이 함수의 분모가 0이므로 가정하지 않는다.

근거: ResNet 노트북 3~4번(전체 셀 11·13). [PyTorch AMP 예제](https://docs.pytorch.org/docs/stable/notes/amp_examples.html), [zero_grad](https://docs.pytorch.org/docs/main/generated/torch.optim.Optimizer.zero_grad.html). 코드 설명 제공 상태이며 독립 이해는 아직 확인하지 않았다.

### 모델 파라미터 개수: `sum(p.numel() for p in model.parameters())`

- `model.parameters()`는 모델에 등록된 weight·bias 등의 Parameter 텐서를 하나씩 꺼내게 한다. `p`는 그 텐서를 가리키는 임시 변수다. `p.numel()`은 number of elements(원소 개수)로 shape의 크기들을 곱한 값이며, 숫자 값 자체를 더하는 연산이 아니다.
- **보충 예시:** `torch.nn.Linear(3,2)`는 weight shape `(2,3)`으로 6개, bias shape `(2,)`로 2개다. 따라서 `sum(...)`은 `6+2=8`을 계산한다. 한 줄은 `num_params=0` 다음 `for p in model.parameters(): num_params += p.numel()`과 같은 뜻이다. 괄호 안은 원소 개수를 차례로 제공하는 generator expression(제너레이터 표현식)이다.
- `print(f'params: {num_params:,}')`의 `f`는 변수 값을 문자열에 넣고, `:,`는 숫자에 천 단위 쉼표를 붙이는 표시 형식이다. 예: 1234567 → `params: 1,234,567`. 변수의 숫자 값은 변하지 않는다.
- 이 코드는 동결된 파라미터도 포함한다. 학습 가능한 것만 세려면 `sum(p.numel() for p in model.parameters() if p.requires_grad)`를 사용한다. BatchNorm의 gamma·beta는 포함하지만 running_mean·running_var 같은 buffer는 포함하지 않는다. 출처: 사용자가 제시한 ResNet 실습 코드, 작은 Linear 예시는 보충 설명.

### BatchNorm2d: 채널별 값 정규화

- **Normalize와 같은가?** `(x-평균)/표준편차`라는 기본 계산은 비슷하다. `torchvision.transforms.Normalize`는 전처리에서 사용자가 제공한 고정 mean/std를 쓰고 학습할 파라미터가 없다. BatchNorm은 모델 내부에서 학습 시 현재 배치의 채널별 통계를 계산하고 gamma/beta도 학습한다(기본 설정). 평가 시에는 누적 통계를 사용한다.
- **정규화 ≠ 정규분포 만들기.** 평균·퍼짐을 조정해도 종 모양 분포로 바뀌지는 않는다. 보충 예시: `[0,0,10,10]`은 자체 평균 5·표준편차 5로 표준화하면 `[-1,-1,1,1]`이다. 평균 0·분산 1이지만 여전히 두 값에만 몰려 있다. Normalize에 주어진 통계가 현재 입력 통계와 다르면 현재 입력의 평균 0·분산 1도 보장하지 않는다.
- **Batch Normalization(배치 정규화)**는 중간 특징값을 정규화하고 학습 가능한 scale·shift로 조정하는 레이어다. `BatchNorm2d(64)`는 채널 64개를 각각 처리한다. 학습 시 `(B,C,H,W)`에서 채널 하나를 고정하고 B·H·W에 걸친 값으로 평균·분산을 계산한다. 채널 전체를 한꺼번에 섞어 평균내는 것이 아니다.
- 식: `정규화값=(x-평균)/sqrt(분산+eps)`, `출력=gamma*정규화값+beta`. **예시:** 채널 하나의 값이 `[1,3,5,7]`이면 평균=4, 분산=5, 표준편차≈2.236이다. 초기 gamma=1,beta=0에서 출력은 약 `[-1.342,-0.447,0.447,1.342]`이고 ReLU를 거치면 `[0,0,0.447,1.342]`다. 이 값은 로컬 PyTorch로 확인했다.
- `affine=True`는 채널별 gamma·beta를 학습한다는 뜻이다. `eps=1e-5`는 0에 가까운 분산에서 나눗셈을 안정화한다. `track_running_stats=True`면 이동 통계를 저장하고, `momentum=0.1`은 새 통계 반영 비율이다. 기본 설정의 train에서는 현재 배치 통계를, eval에서는 저장된 통계를 쓴다. optimizer의 momentum과 구분한다.
- shape는 `(B,64,H,W)` 그대로이며 값만 바뀐다. 입력 전처리 `Normalize(mean,std)`는 제공한 고정 통계를 쓰고, BatchNorm은 모델 중간에서 배치 통계·학습 가능한 파라미터를 사용한다. 정규화 직후 평균·분산을 조정하지만 gamma·beta 적용 후 출력이 항상 평균 0·분산 1인 것은 아니다.

### ReLU 식과 한 층에서 값이 바뀌는 흐름

#### 원본 이미지부터 첫 잔차 블록의 F(x) + x까지

CIFAR용 수정 ResNet18, 배치 1 기준이다. 아래 모든 Conv는 kernel_size=3, stride=1, padding=1이며 shape는 실제 PyTorch 순서 B,C,H,W다.

```text
RGB 이미지: 높이 32 × 너비 32 × 채널 3
    ↓ ToTensor·Normalize·배치로 묶기
(1,3,32,32)
    ↓ 첫 Conv: 3 → 64
(1,64,32,32)
    ↓ BN → ReLU → Identity (maxpool 대체)
x = (1,64,32,32)  ← 첫 잔차 블록의 입력
    │
    ├────────────────────────────┐
    ↓                            │
Conv: 64 → 64                    │
    ↓ (1,64,32,32)               │
BN → ReLU                        │ x 그대로 우회
    ↓ (1,64,32,32)               │
Conv: 64 → 64                    │
    ↓ (1,64,32,32)               │
BN                               │
    ↓                            │
F(x): (1,64,32,32)                │
    └─────────── (+) ◀───────────┘
                  ↓ F(x) + x
             (1,64,32,32)
                  ↓ ReLU
             (1,64,32,32)
```

- 여기서 x는 원본 3채널 이미지가 아니라 첫 Conv·BN·ReLU를 지난 64채널 특징이다. F(x)는 잔차 블록의 두 Conv와 BN·중간 ReLU를 통과한 결과다. 이 블록의 최종 출력은 ReLU(F(x)+x)다.
- 덧셈은 같은 위치끼리 하므로 채널이 128개가 되지 않는다. 예: 한 위치에서 x=2, F(x)=-0.5라면 더한 값과 최종 ReLU 출력은 1.5다. shape가 같아도 특징값은 바뀐다. 이후 채널·해상도가 바뀌는 블록에서는 우회 경로에도 변환을 적용해 shape를 맞춘다.
- 근거: CIFAR 실습의 모델 수정 설정과 ResNet18 첫 BasicBlock 구조. 숫자 덧셈은 보충 예시.

- **ReLU = Rectified Linear Unit(정류 선형 유닛)**. 식은 `ReLU(x)=max(0,x)`이며 각 원소에 따로 적용한다. 음수는 0, 양수는 그대로, 0은 0이다. **예시:** `[[-2,3],[0.5,-1]] → [[0,3],[0.5,0]]`. shape는 유지하고 학습할 가중치가 없는 활성화 함수다.
- 한 층을 단순화하면 `Conv(가중합으로 특징 계산) → BatchNorm(값 정규화·학습 가능한 scale/shift 적용) → ReLU(음수 0)`다. 예를 들어 BN 출력이 위의 2×2 숫자라고 가정하면 ReLU 결과가 오른쪽 행렬이다. 이 숫자는 설명용 가상 값이며 실제 모델 출력 계산 결과가 아니다. ReLU의 꺾인 함수가 비선형성을 제공하여 선형 연산만 쌓는 것보다 복잡한 관계를 표현할 수 있게 한다. 음수가 항상 나쁜 정보라는 뜻은 아니다.
- `inplace=True`는 이 연산에서 입력 텐서 저장 공간의 값을 직접 바꾸는 옵션이다. ReLU의 수식과 출력 shape는 바뀌지 않는다. 출력 채널이 64개여도 모든 특징맵의 각 숫자에 같은 규칙을 적용한다.
- 사용자가 붙인 `Conv7×7, stride=2, MaxPool, fc=1000` 출력은 표준 ImageNet용 ResNet 구조다. 실습의 CIFAR 수정 모델(`Conv3×3, stride=1, maxpool=Identity, fc=10`)과 구분한다. 표준 모델에서 입력 `(1,3,32,32)`를 가정하면 첫 Conv→BN→ReLU는 `(1,64,16,16)`, 그 뒤 MaxPool은 `(1,64,8,8)`이다. BN·ReLU는 공간 크기를 바꾸지 않는다.

### `denorm(images[:16]).clamp(0,1)`의 clamp

- `torch.Tensor.clamp(min,max)`는 값을 지정한 범위 안으로 제한한다. 0보다 작으면 0, 1보다 크면 1, 범위 안의 값은 그대로 둔다. **예시:** `[-0.02,0.3,0.8,1.01] → [0,0.3,0.8,1]`. 전체 숫자를 비례해서 늘이거나 줄이는 정규화가 아니라 경계 밖의 값만 잘라내는 clipping이다.
- 이 코드에서는 `images[:16]`으로 최대 앞 16장을 고르고, `denorm`으로 픽셀 값을 되돌린 뒤, 실수 RGB 이미지를 표시할 때의 0~1 범위에 맞춘다. 정확한 정규화를 그대로 역변환하면 원래 범위로 돌아오지만, 부동소수점 오차 등으로 경계를 조금 벗어날 수 있어 표시 전에 제한한다. 항상 범위 밖 값이 생긴다는 뜻은 아니다.
- shape는 그대로다. `.clamp(...)`는 반환값을 새로 만들어 여기서는 `images_dn`에 담으며, 원본 images를 직접 바꾸는 in-place 연산 `.clamp_(...)`와 다르다. 학습용 정규화 텐서를 무조건 0~1로 자르는 것이 아니라 **역정규화 후 시각화용** 처리라는 점을 구분한다.

### `denorm`: 정규화 역변환과 `(1,3,1,1)`

- `denorm`은 denormalize(정규화 되돌리기)의 줄임말로 작성자가 지은 함수명이다. Normalize가 `z=(원래 값-mean)/std`였다면 원래 값은 `z*std+mean`이다. **예시:** 빨강 픽셀 0.6, mean=0.4914, std=0.2023이면 정규화 값은 약 0.537이고, 역변환하면 다시 약 0.6이다. 그림 표시를 위해 ToTensor 이후의 0~1 값으로 되돌리는 단계이며 0~255 정수로 바꾸는 함수가 아니다.
- `torch.tensor(CIFAR10_MEAN)`은 RGB 평균 3개를 `(3,)` 텐서로 만든다. `.view(1,3,1,1)`은 값은 그대로 두고 축을 배치·채널·높이·너비에 맞춘다. std도 같다. `(1,3,1,1)`은 배치 1개 축·채널 3개·공간 1×1이라는 모양이다.
- 입력이 `(B,3,H,W)`일 때 broadcasting(브로드캐스팅)이 크기 1인 축을 맞춰 계산하므로, 각 이미지의 모든 빨강 픽셀에는 빨강 mean/std, 초록에는 초록 값, 파랑에는 파랑 값을 적용한다. **shape 예시:** `(8,3,32,32) * (1,3,1,1) + (1,3,1,1) → (8,3,32,32)`. 채널별 값을 공간·배치 전체에 적용하는 것이며 입력 이미지를 한 장으로 줄이지 않는다.
- 주석의 `-2.5~2.5`는 정확한 고정 범위가 아니다. 같은 mean/std로 정규화한 픽셀을 그대로 되돌리면 부동소수점 오차를 제외하고 원래 0~1로 복원되며, 임의의 텐서를 자동으로 0~1에 제한하는 연산은 아니다. 이 코드에 `(3,H,W)` 한 장을 넣으면 결과는 `(1,3,H,W)`가 되므로 배치용임을 구분한다. 이 함수의 mean/std는 기본 CPU 생성이므로 GPU 입력에는 같은 device로 맞춰야 한다. 출처: ResNet 실습 셀 7.

### `x0, y0 = train_set[0]`와 shape 출력

- `train_set`은 학습용 Dataset 객체, `[0]`은 첫 번째 샘플 조회다. 이 CIFAR10 Dataset은 `(이미지, 정답 번호)`를 반환한다. 설정한 transform이 조회 과정에서 적용되므로 여기서 이미지는 전처리된 텐서다. `train_set[0]`의 0은 클래스 번호가 아니라 샘플 위치다.
- `x0, y0 = ...`는 반환된 두 값을 나눠 담는 unpacking(언패킹)이다. `x0`에는 이미지 텐서, `y0`에는 클래스 번호가 들어간다. 변수명의 0은 작성자가 붙인 이름일 뿐이며 `image, label`로 바꿔도 된다.
- `x0.shape`는 텐서 각 축의 크기로, 이 예시에서는 `torch.Size([3,32,32])`다. `tuple(x0.shape)`는 이를 `(3,32,32)` 튜플로 바꿔 표시한다. 이미지 값이나 축 순서를 바꾸는 연산이 아니다. C=Channel(채널), H=Height(높이), W=Width(너비). 한 장이라 배치 축은 없다.
- **예시:** `sample=train_set[0]`, `image=sample[0]`, `label=sample[1]`은 위 언패킹과 같은 역할이다. 첫 `[0]`은 Dataset에서 첫 샘플, 두 번째 `[0]`은 그 샘플 쌍에서 이미지를 고른다. `print('Sample image tensor shape (C,H,W):', tuple(x0.shape))`는 설명 문자열과 shape를 공백으로 구분해 한 줄에 출력하며, y0는 이 print에 포함되지 않는다.

### 브라우저 Jupyter: 위·아래 Code/Markdown 셀 추가

- 먼저 셀을 선택하고 `Esc`로 명령 모드에 들어간다. `A`는 위에, `B`는 아래에 셀 추가. `M`은 선택 셀을 Markdown으로, `Y`는 Code로 변경한다. 문자 키는 Shift 없이 누르며 순서대로 입력한다.
- **예시:** 아래에 필기 셀 추가는 `Esc → B → M → Enter`. 위에 코드 셀 추가는 `Esc → A → Y → Enter`. Enter는 편집 시작, Shift+Enter는 코드 실행 또는 Markdown 렌더링이다.
- 위 Markdown은 `Esc → A → M`, 아래 Code는 `Esc → B → Y`. 안 되면 영문 입력 상태와 명령 모드를 확인하고 Help의 Keyboard Shortcuts에서 현재 설정을 확인한다. 출처: [Jupyter Notebook Basics](https://nbclassic.readthedocs.io/en/latest/examples/Notebook/Notebook%20Basics.html).

### U-Net 실습의 패딩

- 사용자 전달: U-Net까지 실습했고, 실습 코드에 패딩이 들어간다. 원본 `04_Unet.ipynb` 셀 7에서 **두 용도**를 확인했다.
- `DoubleConv`의 `Conv2d(kernel_size=3, padding=1)`은 합성곱 전후 H·W를 유지한다. **예시:** 32×32 입력이 stride=1의 3×3 Conv를 지나도 32×32다.
- `Up.forward`의 `torch.nn.functional.pad`는 업샘플 결과를 skip 특징맵의 공간 크기에 맞춘 뒤 채널 방향으로 concat하기 위한 것이다. **예시:** 업샘플 결과 H=6,W=8, skip H=7,W=9면 `[left,right,top,bottom]=[0,1,0,1]`로 패딩해 둘 다 7×9로 맞춘다. 채널 수는 이 패딩으로 바뀌지 않는다.
- 이후 문제는 실습 구현을 기준으로 패딩·concat을 설명한다. 원 논문의 공간 크기가 줄어드는 합성곱·crop 방식과 그대로 혼동하지 않는다. 기존 Vision 모의시험 VISION09~10은 padding=1과 Up의 크기 맞춤 패딩을 이미 포함한다.

### 코드 문제의 import·패키지 이름 표시

- 사용자 요청에 따라 ResNet HTML의 코드 문제 10개와 이론 코드에 필요한 import를 추가하고 패키지 이름을 풀어 썼다. **예시:** `import torchvision` 다음 `torchvision.transforms.Compose(...)`로 작성하면 변환 도구의 소속을 바로 알 수 있다.
- 원본의 `T`는 `import torchvision.transforms as T`, `nn`은 `import torch.nn as nn`, `F`는 `import torch.nn.functional as F`에서 붙인 별명이다. 같은 문서에 대응표를 추가했다. `tfms`는 별도로 만든 변수명으로 패키지 import 이름이 아니다.

### DETR 매칭과 loss · 교안 31~33쪽

- [그림으로 보는 DETR loss](../lessons/0029-day04-detr-matching-loss.html)를 작성했다. 매칭은 종류와 박스 비용을 함께 고려해 전체 합이 최소인 일대일 배정을 찾는 과정이다. 종류를 먼저 확정하고 위치를 나중에 고르는 별개의 선택이 아니다.
- **예시:** 고양이·강아지 정답에 P2·P1을 각각 배정했다면, 둘은 종류+박스 loss로 학습한다. 남은 중복 후보 P3는 no-object 종류 loss만 받는다. 박스 오차는 좌표 L1과 GIoU로 평가한다.
- 실습 `03_DETR.ipynb` 셀 12·18은 사전학습 모델을 불러 예측하는 흐름이다. 여기서 배우는 매칭·loss는 학습 원리이며 정답 없는 추론 때마다 수행하는 것은 아니다. 사용자 설명을 제공한 단계로, 구현 숙달이나 시험 제외를 뜻하지 않는다. [한 장 요약](../reference/day04-detr-loss.html).

### 객체 탐지 평가: Precision·Recall·F1·mAP

- **Precision(정밀도)** = TP/(TP+FP): 모델이 찾았다고 한 것 중 진짜 정답 비율. **Recall(재현율)** = TP/(TP+FN): 실제 정답 중 모델이 찾아낸 비율. TP=True Positive(올바른 탐지), FP=False Positive(잘못된 탐지), FN=False Negative(놓친 정답)다.
- **공통 예시:** 실제 고양이가 10마리이고 모델이 고양이라고 박스 8개를 냈는데 6개만 올바르면 TP=6, FP=2, FN=4. Precision=6/8=75%, Recall=6/10=60%. 객체 탐지의 올바른 박스는 클래스·IoU 기준·일대일 매칭을 만족해야 한다. 같은 고양이에 중복 박스를 내도 모두 정답으로 세지 않는다.
- confidence 기준을 높이면 확실한 탐지만 남겨 Precision이 높아지는 경향이 있지만 놓치는 객체가 늘어 Recall이 낮아질 수 있다. 기준을 낮추면 보통 반대 경향이다. Precision이 매 순간 반드시 단조롭게 변하는 것은 아니며, 좋은 모델은 둘 다 높일 수 있다. **F1-score는 이 문제를 해결하는 학습 방법이 아니라 두 값을 함께 요약하는 평가 지표**다. F1=2PR/(P+R)=2TP/(2TP+FP+FN). 위 예시는 약 66.7%다. 조화평균이라 한쪽이 낮으면 함께 낮아진다.
- **AP(Average Precision, 평균 정밀도)**는 한 클래스에서 confidence 기준을 바꿔 가며 얻는 Precision–Recall 곡선을 요약한 점수다. 직관적으로 곡선 아래 면적이며, 보간·합산 방식은 평가 규칙에 따른다. **mAP(mean Average Precision, 클래스별 AP의 평균)**는 객체 탐지에서 널리 쓰인다. 예: 고양이 AP=.80, 강아지 AP=.60이면 mAP=.70. 한 confidence 기준에서의 Precision 평균이나 F1 평균이 아니다.

| 지표 | 쉽게 묻는 말 | 평가 기준 |
| --- | --- | --- |
| Precision | 찾았다는 것 중 얼마나 맞았나? | 선택한 confidence·IoU 기준 |
| Recall | 실제 정답 중 얼마나 찾았나? | 선택한 confidence·IoU 기준 |
| F1 | 위 두 값이 함께 좋은가? | 선택한 confidence·IoU 기준 |
| AP | 한 클래스에서 confidence 기준을 바꿔도 잘 찾나? | PR 곡선 요약, IoU 기준 명시 |
| mAP | 여러 클래스에서 전반적으로 잘 찾나? | AP의 클래스 평균, IoU 기준 명시 |

**IoU(Intersection over Union, 교집합/합집합)**는 예측 박스와 정답 박스의 겹침 정도다. confidence는 예측을 남길지, IoU는 남긴 예측의 위치가 맞는지 판단하는 서로 다른 기준이다. **mAP@0.5**의 0.5는 IoU 기준이다. COCO 방식 **mAP@0.5:0.95**는 IoU 0.50,0.55,…,0.95의 10개 기준에서도 평균내므로 정확한 박스 위치까지 더 엄격하게 본다. mAP는 객체 탐지의 대표 지표이며 모든 AI 과제에서 항상 가장 많이 쓰는 지표라는 뜻은 아니다.

출처: `vision/04_Applications.pdf` 6~9쪽. 고양이 숫자와 AP 평균은 이해를 위한 보충 예시다. 실제 AP 계산 구현은 해당 평가 도구의 규칙에 맞춘다.

### 강조 코드: train_tfms와 test_tfms

- `T`는 torchvision.transforms의 별명, `tfms`는 transforms(변환들)의 줄임말이다. `Compose`는 변환을 위에서 아래 순서로 적용한다. 학습에는 RandomCrop·RandomHorizontalFlip을 넣어 같은 이미지도 다르게 보여주고, 평가는 이 두 랜덤 변환을 빼서 평가 입력을 일관되게 만든다.
- **예시:** CIFAR의 32×32 RGB 이미지 → RandomCrop의 padding=4로 사방에 기본값 0을 채워 40×40 → 임의 위치에서 32×32 추출 → 기본 확률 0.5로 좌우 반전 → ToTensor `(3,32,32)` → Normalize 같은 shape. DataLoader가 8장을 묶으면 `(8,3,32,32)`다. 두 랜덤 변환 모두 클래스 라벨은 그대로 둔다.
- uint8 PIL 이미지의 `ToTensor`는 채널 우선 float 텐서로 바꾸며 픽셀을 0~255에서 0~1로 조정한다. `Normalize`는 채널별 `(값-mean)/std`를 계산한다. **test에도 ToTensor·Normalize는 필요**하며, 여기서는 train과 같은 mean/std를 사용한다. 정규화 결과가 항상 0~1인 것은 아니다.
- 시험 대비 후보: train/test 변환 구분, Crop의 32·padding의 4, ToTensor 다음 Normalize 순서, 샘플 CHW와 배치 BCHW 구분. Normalize는 매 이미지에서 mean/std를 새로 구하지 않고 주어진 값을 사용한다. 평가 데이터로 정규화 통계를 새로 맞추지 않는다. 출처: 사용자가 제공한 실습 코드 및 기존 ResNet 전처리 실습.

### Zero-shot capability란?

- **Zero-shot capability(제로샷 능력)**는 새 과제의 정답 예제로 별도 학습하지 않고도, 이미 배운 지식을 이용해 그 과제를 수행하는 능력이다. 여기서 zero는 새 과제용 학습 예제가 0개라는 뜻이며, 모델이 아예 학습되지 않았다는 뜻은 아니다.
- **예시:** 사전학습된 CLIP에 고양이 사진과 후보 문구 `a photo of a cat`, `a photo of a dog`, `a photo of a car`를 준다. 이미지와 각 문구의 벡터를 비교해 가장 잘 맞는 문구를 선택한다. 이 분류 과제용 사진·정답으로 추가 학습하는 과정은 없다.
- CLIP은 **Contrastive Language-Image Pre-training(대조 방식의 언어·이미지 사전학습)**의 약자다. 미리 이미지·텍스트 관계를 학습했기 때문에 새 분류 후보를 문장으로 지정할 수 있다. 고양이라는 개념 자체를 사전학습에서 한 번도 보지 않았다는 뜻은 아니다.
- 출처: [03_Attention.pdf](../vision/03_Attention.pdf) 38~39·44·49쪽. 위 고양이 후보 예시는 이해를 위한 보충 예시다. Zero-shot은 수행 방식이며 항상 정답을 맞힌다는 보장은 아니다.

### Few-shot learning이란?

- **Few-shot learning(소수 예제 학습)**은 새 과제의 정답 예제를 몇 개만 이용해 그 과제를 배우거나 적응하는 것이다. Vision의 K-shot 분류에서는 보통 **클래스마다 K개**의 정답 있는 예제를 뜻한다. 모델은 이미 사전학습된 경우가 많다.
- **예시:** 고양이·강아지 분류에서 고양이 사진 4장과 강아지 사진 4장에 정답을 붙여 사용하면 2개 클래스의 **4-shot 분류**다. 총 8장을 사용하고, 별도의 처음 보는 사진으로 평가한다. Zero-shot은 이 과제용 정답 예제를 제공하지 않고, one-shot은 클래스당 1개, few-shot은 몇 개를 제공한다.
- **현재 교안의 방식:** CLIP 이미지 encoder는 동결하고, 그 위의 작은 선형 분류기만 소수 예제로 학습한다. 이를 **few-shot linear probe(소수 예제로 선형 분류기 학습)**라고 부른다. 전체 모델을 반드시 다시 학습하는 것은 아니다.
- 출처: [03_Attention.pdf](../vision/03_Attention.pdf) 47쪽. 보충 구분: LLM에서 말하는 few-shot prompting은 프롬프트에 예시 몇 개를 보여주는 방식이며, 보통 가중치 업데이트 없이 수행한다. 현재 Vision 교안의 분류기 학습과 구분한다.

- 질문·코드 위치:
- 짧은 설명과 예시:
- 오류 원인·수정 결과:

## 오늘 과제와 제출

- **우선 학습 순서 변경(사용자 요청):** [기존 ResNet 9단계 HTML](../lessons/0028-day03-resnet18-shape-journey.html)에서 이론·shape·코드 문제를 먼저 공부하고, 이후 Jupyter에서 모의시험을 푼다. 9단계에 코드 문제 10개와 이유 설명란·힌트·시도 후 해설을 추가했다. 9단계는 학습 루프와 평가 지표 두 문제다.
- **예시:** 전처리 이론에서 train/test 차이를 이해한 뒤 `make_transforms`의 Crop·Flip·ToTensor·Normalize 빈칸을 채우고, 나중에 VISION01을 실제 실행한다. [단계·코드 복습표](../reference/day04-resnet-code-map.html)도 제공했다. 입력은 새로고침하면 사라지며, HTML 확인은 Python 실행이 아니다.

- 오늘 교안에 맞는 기본 과제: 질문·이해도 확인 후 지정. 현재 실습은 U-Net까지 진행했으며 사용자 요청에 따라 ResNet HTML 이론 복습도 이어간다.
- 제출 위치: `practice/day-04/`.
- 기존 모의시험 이어 풀기: [LLM](../practice/day-03/llm-mock-exam.ipynb), [Vision](../practice/day-03/vision-mock-exam.ipynb).
- Vision 문제지 VISION01~12를 수업 실습 복습에 활용할 수 있다. U-Net은 VISION09~11과 연결한다. DETR은 별도 확정된 13절까지의 시험 범위를 계속 적용한다.
- 결과: 미시도. 기존 문제지 제공과 제작자 검증은 학습자의 독립 풀이·숙달을 뜻하지 않는다.
- 받은 힌트·풀이 시간·재시도할 문제:

## 복습 예약

| 시점 | 복습할 내용 | 실제 수행 결과 |
| --- | --- | --- |
| Day 5 · 다음 수업일 | 오늘 배운 핵심·오답, 수업 후 지정 | 미시도 |
| Day 7 · 3수업일 후 | 오늘 개념을 다른 입력·shape로 재확인 | 미시도 |
| Day 11 · 7수업일 후 | 오늘 함수의 빈칸·부분 구현 재시도 | 미시도 |

## 마무리

- [ ] 핵심을 내 말로 설명했다.
- [ ] 빈칸·부분 구현을 자료 없이 시도했다.
- [ ] 오류와 도움받은 부분을 기록했다.
- [ ] 다음 복습 주제를 정했다.
