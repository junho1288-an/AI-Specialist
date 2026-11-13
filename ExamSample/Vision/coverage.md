# PDF 출제 포인트·자가점검과 문제 대응

근거: `AI인증시험_출제포인트.pdf` 1쪽 Vision. 이 표는 연습문제 설계이며 실제 출제를 확정하지 않습니다.

| PDF 자가점검 항목 | 대응 문제 |
|---|---|
| 주요 layer의 의미·구성·동작·활용 | R04, V01~V09, D04~D05, U02~U06 |
| 모델 결과와 정답을 이용하여 학습으로 연결 | R05~R06, V10~V11, U07 |
| 모델·학습·평가 의도에 맞는 데이터 구성 | R01~R03, V01~V02, D01~D03, U01·U08 |
| torchvision.transforms 등의 이미지 변형 | R01·R03, D01 |
| 평가 Metric 원리와 방법 | R07 정확도·샘플 평균 loss, R08 confidence와 정확도 구분 |

범위 안 평가 루프의 top-1 accuracy를 코드로 평가합니다. PDF는 구체적인 지표명을 지정하지 않습니다. ResNet6절 confusion matrix는 이번 범위 밖이고, DETR 실습에는 mAP·Hungarian loss 구현이 없어 코드 문제를 새로 추가하지 않았습니다. U-Net은 BCE loss/이진 예측을 다루며 loss를 IoU·Dice와 같은 지표로 혼동하지 않습니다.

| 문제 | 주제 | 원본 파일 | 전체 셀(1부터) | PDF 연결 |
|---|---|---|---|---|
| R01 | 학습·평가 전처리 구분 | 01_ResNet18_CIFAR10.ipynb | 6 | 데이터 구성·이미지 변환 |
| R02 | DataLoader로 이미지·정답 묶기 | 01_ResNet18_CIFAR10.ipynb | 6 | 데이터 구성 |
| R03 | 역정규화와 broadcasting | 01_ResNet18_CIFAR10.ipynb | 7 | 데이터 구성·이미지 변환 |
| R04 | CIFAR용 ResNet stem과 분류기 | 01_ResNet18_CIFAR10.ipynb | 9 | 레이어 구성·동작 |
| R05 | 분류 loss와 학습할 파라미터 연결 | 01_ResNet18_CIFAR10.ipynb | 13 | 예측·정답을 학습으로 연결 |
| R06 | AMP 학습 한 배치 | 01_ResNet18_CIFAR10.ipynb | 11 | 예측·정답을 학습으로 연결 |
| R07 | 평가 모드·정확도·샘플 가중 평균 | 01_ResNet18_CIFAR10.ipynb | 11 | 평가 Metric·학습/평가 구분 |
| R08 | 추론 확률·클래스·confidence | 01_ResNet18_CIFAR10.ipynb | 15 | 레이어 활용·평가 Metric |
| V01 | 이미지 패치화와 선형 임베딩 | 02_ViT_CIFAR10.ipynb | 11 | 레이어 구성·데이터 구성 |
| V02 | CLS 토큰과 위치 임베딩 | 02_ViT_CIFAR10.ipynb | 11 | 레이어 구성·동작 |
| V03 | PreNorm: 먼저 LayerNorm | 02_ViT_CIFAR10.ipynb | 9 | 레이어 구성·동작 |
| V04 | FFN의 확장·활성화·축소 | 02_ViT_CIFAR10.ipynb | 9 | 레이어 구성·동작 |
| V05 | QKV 분리와 head 축 만들기 | 02_ViT_CIFAR10.ipynb | 9 | 레이어 구성·동작 |
| V06 | Scaled attention과 V 가중합 | 02_ViT_CIFAR10.ipynb | 9 | 레이어 구성·동작 |
| V07 | head 합치기와 출력 투영 | 02_ViT_CIFAR10.ipynb | 9 | 레이어 구성·동작 |
| V08 | 두 residual 연결의 순서 | 02_ViT_CIFAR10.ipynb | 9 | 레이어 구성·동작 |
| V09 | CLS/mean pooling과 클래스 logits | 02_ViT_CIFAR10.ipynb | 11 | 레이어 구성·동작 |
| V10 | log_softmax와 NLLLoss 학습 | 02_ViT_CIFAR10.ipynb | 19 | 예측·정답을 학습으로 연결 |
| V11 | 체크포인트에서 학습 상태 복원 | 02_ViT_CIFAR10.ipynb | 15, 23 | 학습에 필요한 작업·레이어 활용 |
| D01 | DETR 입력 리사이즈·정규화·배치 | 03_DETR.ipynb | 8, 18 | 데이터 구성·이미지 변환 |
| D02 | 중심 박스를 픽셀 꼭짓점으로 | 03_DETR.ipynb | 8 | 데이터 변환·모델 출력 활용 |
| D03 | no-object 제외와 confidence 필터 | 03_DETR.ipynb | 18 | 모델 출력 활용·데이터 구성 |
| D04 | forward hook으로 attention 추출 | 03_DETR.ipynb | 22 | 레이어 활용 |
| D05 | attention 토큰축을 공간으로 복원 | 03_DETR.ipynb | 24, 28, 30 | 레이어 동작·출력 활용 |
| U01 | 이미지와 binary mask의 채널축 | 04_Unet.ipynb | 5 | 데이터 구성 |
| U02 | DoubleConv의 padding과 채널 | 04_Unet.ipynb | 7 | 레이어 구성·동작 |
| U03 | Down: pooling 후 convolution | 04_Unet.ipynb | 7 | 레이어 구성·동작 |
| U04 | 두 업샘플링 방식 | 04_Unet.ipynb | 7 | 레이어 구성·동작 |
| U05 | Up: 크기 정렬과 skip concat | 04_Unet.ipynb | 7 | 레이어 구성·동작 |
| U06 | Decoder에 올바른 skip 전달 | 04_Unet.ipynb | 7 | 레이어 구성·동작 |
| U07 | 픽셀 logits와 정답 mask로 학습 | 04_Unet.ipynb | 9, 11 | 예측·정답을 학습으로 연결 |
| U08 | 확률을 binary segmentation mask로 | 04_Unet.ipynb | 15 | 모델 출력 활용·데이터 구성 |
