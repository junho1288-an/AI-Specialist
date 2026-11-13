# Vision 시험 연습 · 32문제

[전체 문제 노트북](vision-exercises.ipynb)을 열어 준비 셀부터 실행하세요. 한 번에 하나의 노트북만 풀면 됩니다. 아래 개별 파일은 통합본의 같은 문제를 분리한 것입니다.

| 파일 | 문제 | 적용 범위 |
|---|---|---|
| [ResNet](01-resnet-exercises.ipynb) | R01~R08 · 8개 | 0~5절, 전체 셀15까지 |
| [ViT](02-vit-exercises.ipynb) | V01~V11 · 11개 | 6절 이전, 전체 셀23까지 |
| [DETR](03-detr-exercises.ipynb) | D01~D05 · 5개 | 기존 안내대로13절 포함, 전체 셀30까지 |
| [U-Net](04-unet-exercises.ipynb) | U01~U08 · 8개 | 지정 실습 전체 |

원본에 빈칸이 없어 **완성된 코드에서 선정한 연습용 마스킹**입니다. [시험 PDF와 문항 대응표](coverage.md)에 PDF 1쪽 Vision 출제 포인트·자가점검 항목, 원본 파일·전체 셀 번호를 연결했습니다. 실제 시험 문항·배점 확정 안내는 아닙니다.

## 푸는 순서

1. 준비 셀을 실행합니다. 각 문제에 필요한 import도 코드 셀에 표시했습니다.
2. 문제의 입력·반환값·shape를 읽고 `MISSING` 또는 `raise NotImplementedError`를 교체합니다. `# ????`는 표시용 주석입니다.
3. 작성한 함수 셀, 바로 아래 검증 셀 순서로 실행하여 PASS/FAIL을 확인합니다. 처음 빈칸 상태에서는 FAIL이 정상입니다.
4. 시도 후 접힌 해설·예시·정답을 확인합니다. 다음날 해설을 닫고 다시 풀어보세요.

각 문제는 독립적입니다. CPU에서 작은 더미 입력으로 확인하며 CIFAR·COCO·모델 다운로드, 긴 학습, GPU가 필요 없습니다. 정답 검증은 실제 모델의 최종 성능을 평가하지 않습니다. 이미지 shape는 PyTorch의 **B,C,H,W** 순서입니다.

필요 패키지: PyTorch, torchvision, einops, NumPy, Pillow. 개발 검증에는 nbformat·nbclient·ipykernel을 사용했습니다. 현재 `env_aias_test`의 torch2.5.1/torchvision0.20.1/einops0.8.2에서 실행합니다.

## 원본과의 차이·범위 판단

- 이번 요청의 “Chapter 6 이전”을 우선하여 **ResNet6절 자체도 제외**했습니다. 이전의 “6번까지 포함”보다 좁은 이번 자료 범위입니다. ViT6절 Attention 시각화도 제외했습니다.
- ViT FFN: 주석에는 GELU지만 실제 지정 파일은 ReLU라서 정답은 ReLU입니다. U-Net DoubleConv는 실제 코드대로 BN을 넣지 않습니다.
- DETR D05: `(query,key)` attention을 `(Hq,Wq,Hk,Wk)`로 해석합니다. 원본의 반대 방향 슬라이스·축 설명을 교정한 문제임을 명시했습니다.
- 전처리는 실제 torchvision, 패치 변환은 실제 einops를 사용합니다. ResNet은 weights=None이며 다운로드하지 않습니다. 일부 forward 테스트는 작은 모듈을 제공하며 무엇을 대체했는지 문제에 적었습니다.
- 평가 항목은 범위 내 정확도·샘플 가중 평균 loss로 검증합니다. 원본에 없는 mAP/Hungarian matching 구현을 임의로 추가하거나 ResNet6절 confusion matrix를 가져오지 않았습니다. confidence와 accuracy는 구분합니다.
- 설치/다운로드/파일 읽기, 실행시간 측정과 그림 꾸미기는 문제에서 제외했습니다. 체크포인트는 메모리 dict로 제공합니다. ResNet AMP는 비활성 GradScaler로 갱신 순서를 검사하며 GPU 혼합정밀도 수치 검증은 아닙니다.

## 검증·재생성

- `python validate_exercises.py`: 통합·분리본 정답 개별 검사와 새 커널의 빈칸/정답 전체 실행. 학습 노트북을 덮어쓰지 않습니다.
- `python build_exercises.py`: 모든 자료 재생성 및 정답·빈칸·대표 오답 각32개 검증. **작성한 답을 덮어쓰므로 학습 중에는 실행하지 마세요.**
- `validation.json`, `kernel-validation.json`: 검사 결과. `source-manifest.json`: 대조한 원본 SHA-256과 셀 경계.

각 문제는 핵심 코드 단위를 연습하기 위한 것이며 원본의 모든 줄을 시험문제로 바꾼 것은 아닙니다. PASS 외에도 shape·연산 이유를 설명할 수 있는지 별도로 점검하세요.
