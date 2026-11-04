# Day 3 · 01_Intro.pdf 코드 조각과 설명

원문: [01_Intro.pdf](01_Intro.pdf), **PDF 17쪽 / From RGB images To Mini-batch tensor**. 2026-11-04 전체 31쪽의 텍스트를 확인한 결과 Python API가 식별된 페이지는 17쪽이다. 이미지 안의 텍스트에 대한 전 페이지 OCR·육안 검사는 하지 않았다. 다른 Vision PDF는 이 문서의 수집 범위에 포함하지 않았다.

아래 **교안 코드** 네 조각은 17쪽에서 가져왔다. import·변수 대입·실행 예제·shape 설명은 학습을 위해 보완한 내용이다. 코드 조각 자체는 완성된 프로그램이 아니다.

## 코드 목록과 흐름

| 순서 | 교안 코드 | 역할 | 예시 shape |
| --- | --- | --- | --- |
| 1 | `PIL.Image.open(...).convert("RGB")` | 이미지 읽기·RGB 3채널로 변환 | 배열로 표현하면 (H,W,3) |
| 2 | `transforms.ToTensor()` | float 텐서화·0~1 스케일링·채널 축 이동 | (H,W,3) → (3,H,W) |
| 3 | `transforms.Normalize(mean, std)` | 채널별 평균 빼고 표준편차로 나누기 | (3,H,W) 유지 |
| 4 | `tensor.unsqueeze(0)` | 맨 앞에 크기 1인 배치 축 추가 | (3,H,W) → (1,3,H,W) |

```text
RGB 이미지       ToTensor         Normalize        unsqueeze(0)
  H,W,3      →    3,H,W      →      3,H,W      →      1,3,H,W
 0~255            0~1          (값-mean)/std        이미지 1개 배치
```

H=Height(높이), W=Width(너비), C=Channel(채널), B=Batch size(배치 크기). RGB는 Red·Green·Blue(빨강·초록·파랑)라 C=3이다. PIL 이미지 자체는 NumPy 배열·Tensor가 아니며, 위 HWC 표기는 픽셀 데이터를 배열로 표현했을 때의 구조다.

## 1. 이미지 열고 RGB로 맞추기

교안 코드:

```python
PIL.Image.open(...).convert("RGB")
```

실제 경로와 import를 보완하면:

```python
from PIL import Image

with Image.open("sample.png") as source:
    image = source.convert("RGB")
```

- `Image.open(...)`: 로컬 이미지 파일을 연다. `...` 자리에는 파일 경로나 파일 객체를 전달한다.
- `.convert("RGB")`: 흑백·RGBA 등 다른 모드를 RGB 3채널로 맞춘다. RGBA의 알파 채널은 결과에 남지 않는다.
- `from PIL import Image`를 썼으므로 실행 예제에서는 `Image.open`으로 호출한다.
- **예시:** 높이 2·너비 4의 RGB 이미지에는 픽셀 8개가 있고, 픽셀마다 R·G·B 숫자 3개가 있다. PIL의 `image.size`는 **(W,H)=(4,2)** 순서이므로 HWC shape와 구분한다.

## 2. ToTensor: 세 단계를 한 번에

교안 코드:

```python
transforms.ToTensor()
```

적용 예시:

```python
from torchvision import transforms

to_tensor = transforms.ToTensor()  # 변환 도구 생성
tensor = to_tensor(image)         # 도구에 이미지를 넣어 실제 변환

# 한 줄로 쓰면 tensor = transforms.ToTensor()(image)
```

- 이 문서의 **8비트 PIL RGB 이미지**에서는 정수 픽셀을 부동소수 텐서로 바꾸고 255로 나눈다. 기본 float 설정에서는 `torch.float32`다.
- HWC의 채널 마지막 구조를 **CHW의 채널 우선 구조**로 바꾼다. RGB의 숫자 순서를 BGR로 바꾸는 것은 아니다.
- **예시:** 픽셀 `[255,128,0]`은 `[1.0,0.50196,0.0]`이 된다. 높이 2·너비 4 이미지 전체의 텐서 shape는 `(3,2,4)`다.

교안의 2~4단계(실수 변환·스케일링·축 재배치)를 ToTensor 한 번이 처리한다. 다시 `/255`를 하거나 같은 축 변경을 추가하면 안 된다. 이 설명은 8비트 RGB 입력 기준이며, 이미 float인 NumPy 배열 등 모든 입력을 무조건 255로 나누는 API는 아니다. [공식 ToTensor 문서](https://docs.pytorch.org/vision/0.20/generated/torchvision.transforms.ToTensor.html)

## 3. Normalize: 채널마다 값 조정하기

교안 코드:

```python
transforms.Normalize(mean, std)
```

학습용 숫자를 넣은 적용 예시:

```python
normalize = transforms.Normalize(
    mean=(0.5, 0.5, 0.5),  # R, G, B 순서
    std=(0.5, 0.5, 0.5),
)
normalized = normalize(tensor)
```

- 각 채널에 `(픽셀값 - 해당 채널 mean) / 해당 채널 std`를 적용한다. std는 0이면 안 된다.
- 여기의 0.5는 **설명용으로 선택한 값**이다. 교안에 이 숫자들이 지정된 것은 아니며, 실제 모델·전처리 조건이 주어지면 그 값을 따른다.
- **예시:** `[1.0,0.50196,0.0]` → `[1.0,0.00392,-1.0]`. shape는 `(3,2,4)` 그대로이고, 이 설정에서는 0~1이 -1~1로 바뀐다.
- 이 함수가 사진에서 평균·표준편차를 자동 계산하거나, 모든 사진의 실제 평균을 정확히 0으로 만드는 것은 아니다. 전달받은 mean·std로 계산한다. Tensor에 적용하므로 ToTensor 뒤에 둔다. [공식 Normalize 문서](https://docs.pytorch.org/vision/0.20/generated/torchvision.transforms.Normalize.html)

## 4. unsqueeze(0): 이미지 1개를 배치로

교안 코드:

```python
tensor.unsqueeze(0)
```

정규화한 텐서에 적용하는 예시:

```python
batch = normalized.unsqueeze(0)
# (3,2,4) → (1,3,2,4)
```

- `0`은 **새 축을 넣을 위치**다. 0개를 넣거나 픽셀 값을 0으로 채우는 것이 아니다.
- 픽셀 숫자는 유지하고 바깥에 “이미지 1개짜리 묶음” 축을 만든다.
- **예시:** 이미지 한 장 `(3,224,224)` → 한 장 배치 `(1,3,224,224)`. `unsqueeze(0)`만으로 이미지 여러 장이 생기지는 않는다.

배치 B장을 만들려면 같은 shape의 이미지 텐서들을 `torch.stack([image1, image2, ...], dim=0)`으로 모으거나 DataLoader가 묶게 한다. stack·DataLoader는 이 페이지 원문의 네 코드에 포함되지 않은 연결 설명이다. [공식 unsqueeze 문서](https://docs.pytorch.org/docs/stable/generated/torch.unsqueeze.html)

## 연결해서 실행하기: 다운로드 없이

아래는 **원문 네 API를 연결한 보충 예제**다. 외부 파일 대신 메모리에서 RGB 이미지를 만들어 시작한다. 원문 그대로의 완성 코드로 인용한 것이 아니다.

```python
from PIL import Image
from torchvision import transforms

# (너비 4, 높이 2), 모든 픽셀의 RGB는 (255,128,0)
image = Image.new("RGB", (4, 2), color=(255, 128, 0))

tensor = transforms.ToTensor()(image)
normalized = transforms.Normalize(
    mean=(0.5, 0.5, 0.5),
    std=(0.5, 0.5, 0.5),
)(tensor)
batch = normalized.unsqueeze(0)

print(tensor.shape)        # torch.Size([3, 2, 4])
print(tensor[:, 0, 0])     # 첫 픽셀의 RGB: 약 [1.0000, 0.5020, 0.0000]
print(normalized[:, 0, 0]) # 약 [1.0000, 0.0039, -1.0000]
print(batch.shape)         # torch.Size([1, 3, 2, 4])
```

실행 확인: Python 3.12.8, PyTorch 2.5.1, torchvision 0.20.1+cpu, Pillow 11.0.0. 메모리 RGB 이미지의 PNG 저장·Image.open·convert, 값 스케일링, 정규화, 배치 축 추가 및 두 이미지 묶기를 검증했다. 강의용 전체 노트북 실행과 학습자 숙달 확인은 별도다.

## 시험 대비로 확인할 것

시험관이 교안의 코드도 시험에 나온다고 했다는 사용자 안내를 반영한 **복습 포인트**다. 아래 함수는 실제 시험 문제가 아닌 연습용 골격이다.

```python
def prepare_image(image, mean, std):
    # 제공된 PIL 이미지를 RGB로 맞추고
    # ToTensor → Normalize → 배치 축 추가 후 반환하세요.
    pass
```

- 입력: 제공된 PIL 이미지, 채널 3개의 mean·std. 출력: `(1,3,H,W)` 텐서.
- 자료를 닫고 함수 내부를 작성한 뒤 RGB 채널 수·dtype·값 범위·배치 축을 설명한다. 예상 연습 시간 5~10분이며 공식 시험 시간은 아니다.
- 외부 다운로드는 필요 없다. 반환값을 써야 하며 print만 하는 것으로 대신하지 않는다.
- 예시 이미지를 `(W,H)=(4,2)`로 제공받았다면 결과 shape는 `(1,3,2,4)`여야 한다. 다른 이미지 크기로도 확인한다.

출처: [교안 17쪽](01_Intro.pdf#page=17), [Pillow Image.open·convert](https://pillow.readthedocs.io/en/stable/reference/Image.html), 위 torchvision·PyTorch 공식 문서. [Day 3 노트](../class-notes/day-03.md), [시험 출제 기준](../EXAM-SCOPE.md).
