# Day 3 · LLM·Vision 모의 실기 안내

2026-11-04 작성. [출제 포인트 PDF](../../AI인증시험_출제포인트.pdf) 1쪽의 LLM·Vision 항목과 현재 실습 노트북을 바탕으로 재구성했다. **실제 기출이나 출제 확정 문제가 아니며, 배점과 시간은 연습용이다.**

| 문제지 | 구성 | 첫 풀이 시간 가안 |
| --- | --- | --- |
| [LLM 모의시험](llm-mock-exam.ipynb) | 기본 14문항·90점 + Chapter 7 확장 2문항·10점 | 기본 150~210분, 확장 25~40분 |
| [Vision 모의시험](vision-mock-exam.ipynb) | 12문항·100점 | 150~210분 |

빈칸 `????`, 코드 줄 완성, 제공 클래스·함수의 일부 구현으로 구성했다. 모든 문제에 요구사항, 인자, 입력·출력 shape, 교안 파일·셀 번호, 공개 확인 셀, 이유 설명란이 있다. 각 문제는 자기 노트북의 공통 준비 셀만 실행하면 독립적으로 풀 수 있다.

## 실행과 제출

1. 브라우저 Jupyter Notebook에서 문제지를 열고 복사본을 만든다. `llm-mock-exam-submission.ipynb`, `vision-mock-exam-submission.ipynb`로 저장한다.
2. PyTorch가 설치된 `env_aias_test` 커널을 선택한다. Vision에는 torchvision과 Pillow도 필요하다. 현재 검증 환경에는 설치되어 있다.
3. 공통 준비 셀을 실행한 뒤 한 문제씩 답안 셀과 확인 셀을 실행한다.
4. `????`는 실제 Python 문법이 아니므로 채우기 전 해당 셀은 SyntaxError가 난다. `raise NotImplementedError`는 해당 줄을 지우고 필요한 본문으로 바꾼다.
5. 확인 셀은 수정하지 않는다. 막힌 문제는 번호·오류를 남기고 다음 문제로 이동한다.
6. 이유·shape·도움 사용 여부를 작성한다. 모두 풀었으면 새 커널에서 처음부터 전체 실행하고 제출한다.

**예시:** `keys.transpose(????, ????)`를 채울 때 메서드 이름만 기억하는 대신 입력이 `(B,T,D)`이고 원하는 결과가 `(B,D,T)`라는 조건으로 판단한다.

설명 자료·정답은 먼저 열지 않는 연습을 권한다. 정답 코드는 문제지에 없다. 공개 확인 결과의 예상 값은 문제 조건을 검증하기 위한 예시이며 하드코딩해 반환하면 요구 충족으로 인정하지 않는다.

## PDF 체크포인트와 문항 대응

| PDF 분야·체크포인트 | 대응 문항 |
| --- | --- |
| LLM: 주요 layer의 의미·동작·구성·활용 | LLM02 임베딩, 03 causal attention, 04 multi-head, 05 정규화·FFN, 06 residual, 07 GPT, 13 LoRA |
| LLM: 입력부터 출력까지 Tensor 흐름 | LLM01~09, 11~12, 15 |
| LLM: 무엇을 학습하며 어떤 작업이 필요한가 | LLM05 파라미터, 09 loss, 10 학습, 12 동결·head, 13 LoRA, 16 DPO |
| LLM: 학습·평가 목적에 맞는 데이터 구성 | LLM01 next-token, 11 분류 패딩, 14 분류 평가, 15 instruction collate |
| Vision: 주요 layer의 의미·구성·동작·활용 | VISION03 ResNet, 06~08 ViT, 09~10 U-Net, 12 DETR 출력 활용 |
| Vision: 예측과 정답을 학습으로 연결 | VISION04 분류 학습·평가, 11 segmentation 손실 |
| Vision: 학습·평가 의도에 맞는 데이터 구성 | VISION01 증강·평가 변환, 02 배치, 11 픽셀 정답 |
| Vision: torchvision.transforms 등 이미지 변환 | VISION01, 06 패치 구성, 12 박스 좌표 변환 |
| Vision: 평가 Metric의 원리·방법 | VISION04 전체 accuracy·샘플별 평균 loss, 05 confusion matrix·클래스별 정답 비율 |

표는 PDF의 넓은 평가 관점을 실제 코드 문제로 옮긴 출제 설계다. PDF가 각 모델·문항의 출제를 개별적으로 확정했다는 뜻은 아니다.

## 범위와 원본에서 바꾼 부분

- [EXAM-SCOPE.md](../../EXAM-SCOPE.md)에 따라 Chapter 2의 **Tokenizer 비교부터 이후 내용**, 온라인 다운로드·HTTP·모델 다운로드 절차는 제외했다.
- 주된 LLM 대비는 Chapter 2 비교 이전·Chapter 3~6이다. 폴더에 있는 Chapter 7 지시 학습·DPO는 확장 2문항으로 구분했다. 기존 20시간 계획을 따를 때는 기본 14문항을 먼저 푼다. 기본만 채점하면 90점 만점이며 100점 환산도 별도로 표시한다.
- Vision은 네 노트북 모두에서 문제를 냈다. Intro PDF 17쪽의 RGB·ToTensor·Normalize·배치 축 추가도 연결했다. 파일 열기 자체는 제공 PIL 이미지로 대체했다.
- CPU에서 빠르게 풀도록 합성 텐서와 작은 모델을 사용한다. 전체 CIFAR 학습, 학습된 GPT·DETR 다운로드, 목표 정확도 달성은 요구하지 않는다.
- 토큰화 API 확인용 TinyTokenizer는 **BPE가 아닌 간단한 대역**이다. 실제 GPT-2 토큰 ID나 병합 규칙의 정답으로 사용하면 안 된다.
- ViT의 einops 패치 변환은 같은 패치 순서를 갖는 제공 PyTorch 코드로 대체했다. attention의 einsum은 같은 의미의 행렬 곱으로 바꿨다. 분류 학습은 AMP 없는 CPU 경로로 축소했다.
- Vision 코드와 shape 채점은 PyTorch의 실제 `(B,C,H,W)`를 사용한다. 강의 그림의 B,C,W,H 표기와 구분한다. PIL 이미지 크기만 `(W,H)`다.
- 교안의 모든 셀을 한 번씩 출제한 문제집은 아니다. Grad-CAM·시각화 hook·ONNX·모델 저장/복원 등은 이번 세트에서 생략했으며, **시험 제외로 확정한 것이 아니다.**

## 채점과 오답 복습

문항별 점수는 코드 요구 충족 80%, 이유·shape 설명 20%를 기준으로 수동 채점한다. 공개 확인 셀은 자동 점수 계산기가 아니다. 제공 사례 통과와 다른 인자에서도 요구 충족, 독립 풀이, 다음날 재현은 구분한다. 공식 합격선을 임의로 정하지 않는다.

틀린 문제는 API·축/shape·데이터 구성·학습 순서 중 무엇이 원인인지 기록한다. 다음 수업일에는 그 문제의 입력 크기·배치 수를 바꿔 다시 풀고, 3·7수업일 뒤에도 핵심 부분을 복습한다. 실제 제출 전에는 숙달로 표시하지 않는다.

## 제작 검증

- 환경: Python 3.12.8, PyTorch 2.5.1, torchvision 0.20.1+cpu, CPU.
- 제작자가 빈칸을 채운 검증용 코드로 28문항의 공개 확인 셀을 실제 실행했다. 전체 순서 실행과 각 문제의 독립 실행을 확인했다.
- 값·shape, causal mask, gradient 초기화·갱신, 평가 모드, 패딩·EOS 처리, DETR 필터·좌표 등 공개 조건을 확인했다.
- 학생용 노트북에는 빈칸을 유지하고 정답 코드·실행 결과를 넣지 않았다. nbformat 구조와 로컬 자료 링크를 검사했다.
- 브라우저 UI에서 실행한 검증은 아니다. Python에서 셀 코드를 실행했으며 원본 교안의 전체 학습을 실행하거나 변경하지 않았다.

교안 출처는 각 문제의 링크·셀 번호를 참고한다. LLM 원본 실습은 Sebastian Raschka의 *Build a Large Language Model From Scratch* 기반 자료이며, Chapter 5~7 원본 셀에는 Copyright (c) Sebastian Raschka / Apache License 2.0 표기가 있다. 문제의 골격·변수명은 해당 실습에서 교육용으로 재구성했다.
