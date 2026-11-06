# Project
본 폴더는 AI specialist수업을 들으면서 나오는 산출물을 저장하고 daily로 github branch에 반영을 목적으로 한다.

# AI-Specialist
## AVD 사용
link : swschool.net
id : swuser-k23
## 인증 평가 내용
실기는 코드 작성만
jupyter 노트북에서
100% 교안에서만 출제 예정

## 학습 자료 바로가기

- [Day 4 ResNet 실습 0~8번 줄별 해설 노트북](practice/day-04/resnet18-line-by-line.ipynb) — shape·잔차 연결 그림·학습 루프·작은 실행 예제. 시험 범위는 6번까지, 7~8번은 추가 이해용.
- [Chapter 2 HTML 수업 목차](lessons/index.html)
- [Chapter 3 Attention 셀별 수업](lessons/chapter-3.html)
- [Chapter 4 GPT 셀별 수업](lessons/chapter-4.html)
- [매일 필기 가이드](class-notes/README.md) · [첫날 필기](class-notes/day-01.md)
- [Anaconda Prompt 실행 가이드](CMD-JUPYTER-TUTORIAL.md)
- [12일 학습 계획](STUDY-PLAN.md)

HTML 수업은 저장소를 받은 뒤 파일을 브라우저에서 열면 됩니다. 스타일과 상호작용을 위해 `assets/` 폴더도 함께 유지하세요.

## 교안과 함께 내려받기

`llm_hands_on/`은 강의 원본 저장소를 연결한 Git 서브모듈입니다.

```bat
git clone --recurse-submodules https://github.com/junho1288-an/AI-Specialist.git
```

이미 이 저장소를 받은 경우에는 저장소 폴더에서 다음을 실행합니다.

```bat
git submodule update --init --recursive
```

Python 가상환경과 패키지는 각 PC에서 실행 가이드에 따라 준비합니다.
