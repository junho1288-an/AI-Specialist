# NGCF 추천시스템 원본 빈칸

[문제 노트북](recsys-gcf-exercises.ipynb): 원본 빈칸8곳을5문제로 수록했습니다. `RecSys_GCF_practice.ipynb`가 문제, `RecSys_GCF_sol.ipynb`가 정답입니다. NCF 파일은 요청 범위에 포함하지 않았습니다.

- 준비 셀 실행 후 MISSING/NotImplementedError를 답으로 바꾸고 아래 테스트를 실행하세요.
- CPU PyTorch만 사용합니다. MovieLens·torch-geometric·모델 다운로드는 없습니다.
- 원본 제공 코드의 index_add 반환값 유실을 보정했습니다. index_add는 비인플레이스이므로 반환값을 저장하거나 index_add_를 써야 합니다.
- 원본 zeros_like 누적 버퍼는 입력/출력 차원이 같아야 합니다. 제공 레이어는 W1 출력으로 누적하여 다른 차원도 검사합니다. 원본 빈칸 답 자체는 solution을 따릅니다.
- 같은 아이템으로 향하는 여러 엣지의 합산, user/item 경계, concat 축, 비대칭 입력을 테스트합니다. 실제 추천 품질은 검사하지 않습니다.
- [원본/정답 비교](recsys-source-comparison.diff). 재생성은 `python build_recsys_exercises.py` (nbformat 필요). 학습 답안을 덮어쓰므로 작성 중 실행하지 마세요.

| 문제 | 원본 셀 | 빈칸 | 주제 |
|---|---|---|---|
| 1 | 14 | 1 | 엣지별 degree 정규화 |
| 2 | 14 | 2 | 아이템으로 전달하는 메시지 |
| 3 | 14 | 1 | 아이템 메시지 누적과 자기 정보 |
| 4 | 15 | 1 | NGCF 레이어 순서대로 호출 |
| 5 | 15 | 3 | 레이어 특징 결합과 user/item 분리 |
