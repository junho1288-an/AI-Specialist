# Day 4 실습 · 2026-11-05

오늘 새로 작성하는 노트북과 과제 답안을 저장하는 폴더다. 분야는 Vision이며 이론은 [03_Attention.pdf](../../vision/03_Attention.pdf) 30쪽 Swin Transformer부터 시작한다.

실습은 사용자 전달 기준 [U-Net](../../vision/04_Unet.ipynb)까지 진행했다. 실습의 Conv padding=1과 Up의 공간 크기 맞춤 패딩을 복습에 반영한다. 실습 진행과 독립 구현 숙달은 구분한다.

- 필기·질문: [Day 4 노트](../../class-notes/day-04.md)
- **ResNet 0~8번 줄별 해설:** [resnet18-line-by-line.ipynb](resnet18-line-by-line.ipynb). 원본 코드의 389개 비어 있지 않은 줄을 설명하고, 흐름 그림·작은 실행 예제 9개를 담았다. 7~8번은 시험 범위 밖의 추가 이해용이다. 실행 결과·그림도 저장했다. 원본 코드는 읽기용이며 실행 셀은 합성 데이터만 사용한다.
- 기존 문제지: [LLM 모의시험](../day-03/llm-mock-exam.ipynb), [Vision 모의시험](../day-03/vision-mock-exam.ipynb)
- 모의시험 사용법·출제 근거: [안내](../day-03/mock-exam-guide.md)

Vision 문제지 VISION01~12를 복습에 활용하며, U-Net은 VISION09~11과 연결한다. DETR의 시험 범위는 노트북 13절까지라는 제한을 유지한다.

오늘 모의시험을 풀면 문제지를 이 폴더에 복사해 `llm-mock-exam-submission.ipynb` 또는 `vision-mock-exam-submission.ipynb`로 저장한다. 위 이름은 제출 파일명 예시이며 아직 답안 파일을 생성한 것은 아니다.

일반 실습 파일명 예시: `주제-practice.ipynb`. 교안 원본을 그대로 두고, 필요한 셀을 복사한 뒤 날짜·문제 번호·받은 힌트를 기록한다.

환경은 기존 `env_aias_test` 커널을 사용한다. 빈칸과 부분 구현을 완성한 뒤 확인 셀을 실행하고, 제출 전 새 커널에서 전체 실행한다.
