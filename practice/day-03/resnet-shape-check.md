# Day 3 · ResNet18 단계별 shape 확인

[그림 수업](../../lessons/0028-day03-resnet18-shape-journey.html)을 한 단계씩 진행하며 해당 문제만 먼저 푼다. 연습용이며 실제 문항·배점·시간을 확정한 것이 아니다. 다운로드·전체 학습 없이 구현한다.

## 1단계부터 구두 확인

1. RGB의 3은 무엇인가? CIFAR 이미지 한 장의 배열 관점 HWC shape는?
2. ToTensor와 Normalize를 마친 이미지의 shape는? 둘 중 축 순서를 바꾸는 함수는?
3. 이미지 4장을 모으면 images와 labels의 shape는 각각?
4. Conv2d(3,64,3,stride=1,padding=1)의 출력 shape는? 필터 개수와 클래스 개수의 차이는?
5. 같은 shape끼리 skip 덧셈하면 채널 수가 두 배가 되는가?
6. layer3와 layer4의 출력 shape는? 배치는 왜 유지되는가?
7. 평균 pooling 직후와 flatten 직후의 shape는?
8. Linear(512,10)의 출력은 왜 (B,10)인가?
9. softmax와 argmax의 shape는 각각? CrossEntropyLoss의 입력은?

답을 이 문서에 적거나 채팅으로 제출한다. 정답은 먼저 시도한 뒤 [복습표](../../reference/day03-resnet18-shapes.html)에서 확인한다.

## 함수 내부 구현 연습 · 약 10~15분

제출 위치 권장: `practice/day-03/resnet-shapes.ipynb`. 노트북 파일은 아직 생성하지 않았다. 아래에서는 torch·torch.nn이 import되어 있고, 이미지는 제공된다고 가정한다.

### A. 이미지당 특징 벡터 만들기

```python
def pool_and_flatten(features):
    # features: float 텐서 (B,C,H,W)
    # 채널마다 공간 평균을 내고 (B,C)로 반환하세요.
    pass
```

제공 테스트 입력: `torch.arange(32, dtype=torch.float32).reshape(2,4,2,2)`.

반환 shape는 (2,4), 첫 이미지 첫 채널 평균은 1.5여야 한다. B·C·H·W를 특정 값으로 고정하지 말고 다른 크기도 처리한다.

### B. 클래스 예측하기

```python
def classify_logits(logits):
    # logits: float 텐서 (B,K)
    # 클래스 축에 softmax를 적용하세요.
    # probs, pred를 이 순서로 반환하세요.
    pass
```

제공 테스트 입력: `torch.tensor([[0.,0.,0.,2.,0.,1.,0.,0.,0.,0.], [2.,0.,0.,0.,0.,0.,0.,0.,0.,0.]])`.

probs shape는 (2,10), pred shape는 (2,), 예측 번호는 [3,0]이어야 한다. 각 행의 확률 합이 1인지 확인한다.

### C. 분류 손실 연결하기

```python
def classification_loss(logits, labels, criterion):
    # criterion에 필요한 입력을 전달하고 손실을 반환하세요.
    pass
```

logits는 위 입력, labels는 `torch.tensor([3,0], dtype=torch.long)`, criterion은 `torch.nn.CrossEntropyLoss()`를 제공한다. 반환 shape는 scalar `()`이어야 한다. softmax를 어디에서 적용해야 하는지 이유를 설명한다.

완료 기준: 새 커널에서 실행, 반환 shape·값 확인, 다른 B·C·K에서도 동작, 각 단계 이유 설명. 다음 수업일과 3·7수업일 후 재시도한다. 전체 모델을 외우는 대신 제공 인자와 함수 내부 흐름을 연습한다.
