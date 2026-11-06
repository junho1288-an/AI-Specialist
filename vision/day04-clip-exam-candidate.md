# Day 4 · CLIP 코드 시험 대비 후보

사용자가 CLIP 코드의 시험 후보 추가를 요청했다(2026-11-05). **실제 출제가 확정됐다는 뜻은 아니다.**

페이지 확인: 사용자는 32쪽이라고 지칭했지만, 현재 [03_Attention.pdf](03_Attention.pdf)의 **32쪽은 Swin 구조**, **42쪽 Training (1/2)에 CLIP 코드 그림**이 있다. 아래는 42쪽 코드를 기준으로 정리했다. 43쪽은 contrastive loss, 44쪽은 zero-shot 추론이다.

## 무엇을 하는 코드인가?

**이미지와 그 이미지의 설명을 같은 벡터 공간으로 옮긴 뒤, 올바른 짝의 점수를 높이도록 학습한다.**

CLIP = **Contrastive Language-Image Pre-training**, 대조 방식의 언어·이미지 사전학습.

```text
이미지 → 이미지 encoder → 특징 → 투영·정규화 ┐
                                           ├→ 모든 이미지·문장 쌍의 점수 → 양방향 loss
문장   → 텍스트 encoder → 특징 → 투영·정규화 ┘
```

## 교안 코드 → PyTorch로 읽기

교안 그림은 **NumPy 스타일 의사 코드**다. `l2_normalize`, `cross_entropy_loss`는 그 그림에 정의되지 않았으므로 그대로 실행되는 완성 코드가 아니다. 아래는 핵심 연산을 PyTorch로 옮긴 **보충 예제**다. encoder가 계산한 특징부터 입력받아 다운로드 없이 연습한다.

```python
import torch
import torch.nn.functional as F

def clip_contrastive_loss(image_features, text_features, W_i, W_t, t):
    # 같은 index의 이미지와 문장이 정답 짝인 배치
    I_e = F.normalize(image_features @ W_i, p=2, dim=-1)
    T_e = F.normalize(text_features @ W_t, p=2, dim=-1)

    logits = (I_e @ T_e.T) * t.exp()
    labels = torch.arange(logits.shape[0], device=logits.device)

    image_to_text_loss = F.cross_entropy(logits, labels)
    text_to_image_loss = F.cross_entropy(logits.T, labels)
    loss = (image_to_text_loss + text_to_image_loss) / 2
    return loss, logits
```

인자 조건: `image_features(N,D_i)`, `text_features(N,D_t)`, `W_i(D_i,D)`, `W_t(D_t,D)`, scalar 텐서 `t`. N≥1이며 입력은 같은 device의 호환되는 실수 dtype을 사용한다. 학습하려는 값은 파라미터로 등록하고 optimizer에 포함해야 한다. 이 함수 자체가 파라미터 등록·갱신을 하는 것은 아니다.

| 코드·단계 | 쉬운 뜻 | 출력 shape |
| --- | --- | --- |
| `image_features @ W_i` | 이미지 특징을 공통 D차원으로 바꿈 | `(N,D)` |
| `text_features @ W_t` | 문장 특징도 같은 D차원으로 바꿈 | `(N,D)` |
| `F.normalize(..., dim=-1)` | 각 벡터의 L2 길이를 맞춰 방향을 비교함 | `(N,D)` |
| `I_e @ T_e.T` | 모든 이미지와 모든 문장의 코사인 유사도 | `(N,N)` |
| `* t.exp()` | 점수 차이의 크기를 조절함 | `(N,N)` |
| `arange(N)` | 정답은 같은 번호: 0,1,…,N−1 | `(N,)` long |
| `CE(logits)` / `CE(logits.T)` | 이미지→문장 / 문장→이미지 두 방향으로 정답 짝 찾기 | 각각 scalar |

`t`는 교안의 학습 가능한 스케일 관련 값이다. 코드에서 `exp(t)`가 곱해지므로 **log 스케일**로 읽는다. 온도 τ로 표현하면 `exp(t)=1/τ`에 해당한다. `t` 자체를 그대로 온도라고 보고 나누는 코드로 바꾸지 않는다.

## 예시 하나: 고양이·강아지 두 쌍

배치 순서가 이미지 `[고양이, 강아지]`, 문장 `[고양이 설명, 강아지 설명]`이라고 하자. 투영·정규화된 벡터를 양쪽 모두 `[[1,0],[0,1]]`, `exp(t)=2`로 둔 단순 예시다.

| 이미지 ↓ / 문장 → | 고양이 설명 | 강아지 설명 |
| --- | ---: | ---: |
| 고양이 | **2** | 0 |
| 강아지 | 0 | **2** |

정답 `labels=[0,1]`은 **고양이 이미지의 정답 문장은 0번, 강아지는 1번**이라는 뜻이다. 이는 고정된 고양이·강아지 클래스 ID가 아니라 **현재 배치에서 짝의 위치**다. 문장 순서가 뒤집히면 정답도 `[1,0]`이 되어야 한다.

이 예시에서 한 행의 정답 확률은 약 0.881, 양방향 평균 loss는 약 0.127이다. 학습은 정답 짝과 다른 짝을 더 잘 구분하도록 encoder·투영·스케일의 파라미터를 갱신한다.

## 시험 후보로 기억할 빈칸·부분 구현

1. **투영과 정규화:** `F.normalize(features @ W, p=2, dim=????)`.
2. **모든 짝 점수:** `(I_e @ ????) * t.exp()`와 `(N,N)` shape 설명.
3. **정답 위치:** `torch.????(N)`과 정답이 대각선인 조건 설명.
4. **양방향 손실:** 원래 logits와 전치한 logits로 CE를 구해 평균하기.
5. **학습 연결:** loss 계산 전 gradient 초기화, 이후 backward와 optimizer 갱신.

교안의 `cross_entropy_loss(..., axis=0/1)`를 PyTorch의 동일한 인자로 복사하지 않는다. PyTorch `F.cross_entropy`에는 그 `axis` 인자가 없으므로, 위처럼 **logits 전치로 비교 방향을 바꾼다**. CE에 전달하기 전 softmax를 별도로 적용하지 않는다.

**학습과 zero-shot 추론 구분:** 이 코드는 정답 이미지·문장 쌍으로 CLIP을 학습하는 코드다. 이미 학습한 CLIP으로 zero-shot 분류할 때는 N개 이미지와 C개 후보 문장을 비교하여 `(N,C)` 점수를 얻는다. 그때는 이 학습용 대각선 정답·loss가 필수 과정이 아니다.

검증: 보충 함수는 로컬 PyTorch 2.5.1 CPU에서 위 수치 예시, 서로 다른 이미지·텍스트 특징 차원, backward를 확인했다. 실제 encoder나 전체 CLIP 모델 학습을 실행한 것은 아니다.
