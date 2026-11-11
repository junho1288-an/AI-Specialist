# Data 원본 빈칸 모음

- [시계열 문제](timeseries-exercises.ipynb): 7문제, 아래 안내 참고.
- [NGCF 추천시스템 문제](recsys-gcf-exercises.ipynb): 원본 빈칸 8곳을 5문제로 구성. [추천시스템 안내](RECSYS-README.md).

## 시계열

[통합 노트북](timeseries-exercises.ipynb)을 열어 준비 셀 → 문제 코드 → 검증 코드 순서로 실행하세요. MISSING 또는 NotImplementedError 부분을 완성합니다. 정답은 문제 아래 접혀 있습니다.

- 문제: `data-class/3_4-ts-practice/ts_practice.ipynb`
- 정답: `data-class/3_4-ts-practice/ts_solution.ipynb`
- copy 파일은 정답 기준에서 제외했습니다. solution에 새로 만들었던 추가 문제도 제외했습니다.
- 원본의 미완성 셀24,28,30,35,44를7문제로 구성했습니다. [문제/정답 비교](source-comparison.diff).
- CPU PyTorch, NumPy, scikit-learn만 사용합니다. 데이터 다운로드·GPU·긴학습·그래프는 없습니다.
- Decoder FC는 정답 파일의 `(hidden_size,input_size)`를 따르며 단변량·다변량을 검사합니다.
- MAPE의 .1은10%입니다. 정답0인 샘플은 이 최소 테스트에서 제외합니다.
- 각 문제를 두 입력으로 검증합니다. 모델 출력·shape·gradient·두 배치 업데이트를 검사하며 실제 예측 품질 평가는 아닙니다.
- 재검증: `python validate_exercises.py` (nbformat, nbclient 필요).
- 재생성: `python build_exercises.py`. 학습 중 실행하면 작성한 답을 덮어쓰니 주의하세요.

| 문제 | 구분 | 원본 전체 셀 | 주제 |
|---|---|---|---|
| 01 | 원본 빈칸 | 24 | 한 시점 예측: 학습 배치 |
| 02 | 원본 빈칸 | 24 | 한 시점 예측: 평가 배치 |
| 03 | 원본 빈칸 | 28 | RMSE와 MAPE |
| 04 | 원본 빈칸 | 30 | Conv1DModel 구성과 forward |
| 05 | 원본 빈칸 | 35 | RNNModel 구성과 forward |
| 06 | 원본 빈칸 | 44 | EncoderRNN 구성과 forward |
| 07 | 원본 빈칸 | 44 | DecoderRNN 구성과 forward |
