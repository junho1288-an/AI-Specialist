# Anaconda Prompt에서 수업 노트북 실행하기

사용자가 2026-11-02에 실행에 성공했다고 알려 준 절차를 기준으로 정리했다. 사용 환경은 Conda의 `env_aias_test`, 교안 위치는 `C:\llm_hands_on`이다. 기존 안내의 `C:\AI-Specialist\llm_hands_on`과 다른 경로이므로 실제로 성공한 경로를 사용한다.

실행 명령 `jupyter lab`은 **JupyterLab**을 켠다. 그 안에서 수업의 Jupyter Notebook 파일(`.ipynb`)을 열어 실습한다.

## 1. Windows 검색에서 Anaconda Prompt 실행

시작 버튼 또는 Windows 검색창에서 `Anaconda Prompt`를 검색해 실행한다.

처음에는 다음처럼 표시될 수 있다.

```text
(base) C:\Windows\System32>
```

`(base)`는 현재 선택된 Conda 환경, 그 뒤 경로는 현재 폴더다. 명령어를 입력할 때 이 표시까지 복사하지 않는다.

## 2. 사용할 환경 확인

```bat
conda env list
```

사용자가 확인한 목록:

```text
base                 * C:\ProgramData\miniconda3
env_aias_test           C:\ProgramData\miniconda3\envs\env_aias_test
```

`*`는 현재 활성 환경을 뜻한다. 목록에 `env_aias_test`가 있으면 새로 만들 필요가 없다.

## 3. 수업용 Conda 환경 활성화

```bat
conda activate env_aias_test
```

명령 줄 앞부분이 `(base)`에서 `(env_aias_test)`로 바뀌면 활성화된 것이다.

Python 버전과 실행 경로도 확인할 수 있다.

```bat
python --version
python -c "import sys; print(sys.executable)"
```

예상 실행 경로는 `C:\ProgramData\miniconda3\envs\env_aias_test\python.exe`다. 교안의 요구 버전은 3.12이므로 버전은 `3.12.x`인지 확인한다. 사용자가 제공한 로그에는 이 환경의 Python 버전이 없어 아직 확인된 것으로 기록하지 않는다.

이 절차에서는 Conda 환경을 사용하므로 기존 `.venv` 활성화 명령은 추가로 실행하지 않는다.

## 4. 교안 폴더로 바로 이동

```bat
cd /d C:\llm_hands_on
dir
```

`C:\Windows\System32`에서 `cd ..`를 여러 번 할 필요 없이 절대 경로로 바로 이동할 수 있다. `cd /d`는 드라이브도 함께 바꾼다.

명령 줄이 다음처럼 표시되고 `dir` 결과에 교안 파일들이 보이면 된다.

```text
(env_aias_test) C:\llm_hands_on>
```

- `dir`: 현재 폴더의 파일·폴더 목록 보기. 이 CMD 기반 Anaconda Prompt에서는 `ls` 대신 사용한다.
- `cd ..`: 부모 폴더로 한 단계 이동.
- `cd ...`: 부모 폴더 이동 명령이 아니다.
- 실제 교안 위치가 다르면 `cd /d` 뒤의 경로를 바꾼다.

## 5. Jupyter 실행

```bat
jupyter lab
```

사용자가 제공한 로그에는 다음과 같은 확장 연결 메시지가 나타났다.

```text
jupyter_lsp | extension was successfully linked.
jupyter_server_terminals | extension was successfully linked.
```

이는 시작 과정의 정보 메시지다. 이어서 브라우저가 열리거나 접속 주소가 출력되는지 확인한다. 브라우저가 자동으로 열리지 않으면 Anaconda Prompt에 출력된 실제 `http://localhost:.../lab?token=...` 주소를 복사해 브라우저에서 연다.

서버가 실행되는 동안 Anaconda Prompt 창을 열어 둔다.

## 6. 노트북 열고 테스트

1. 브라우저의 파일 목록에서 교안 `.ipynb`를 열거나, Launcher의 Notebook에서 Python 커널을 선택해 새 노트북을 만든다.
2. 새 테스트 노트북의 셀에 다음 코드를 입력한다.

```python
import sys

print("Python 버전:", sys.version)
print("실행 경로:", sys.executable)
print("계산 결과:", 1 + 2)
```

3. `Shift+Enter`로 실행한다.
4. 계산 결과가 `3`이고 실행 경로가 `env_aias_test\python.exe`를 가리키는지 확인한다. 다른 환경이면 노트북의 커널 선택 메뉴에서 변경한다.
5. `Ctrl+S`로 저장한다. 새 테스트 파일은 `environment-check.ipynb`처럼 이름을 붙인다.
6. 커널을 재시작하고 전체 셀을 다시 실행해 같은 결과가 나오는지 확인한다.

서버 시작과 교안 전체 실행은 별도 확인이다. 교안에서 패키지 오류가 나면 노트북 이름, 실행 경로와 오류 메시지를 함께 남긴다.

## 7. 종료

노트북 저장 → 서버를 켠 Anaconda Prompt에서 `Ctrl+C` → 종료 확인이 나오면 `y`와 Enter를 입력한다. 종료 후 환경을 해제한다.

```bat
conda deactivate
```

브라우저 탭을 닫는 것만으로 서버가 종료되지는 않는다.

## 다음날 실행할 세 줄

Windows 검색에서 Anaconda Prompt를 열고 다음을 실행한다.

```bat
conda activate env_aias_test
cd /d C:\llm_hands_on
jupyter lab
```

## 다른 PC에서 환경이 없다면

아래는 성공 로그에 포함되지 않은 새 환경 설치용 절차다. 기존 환경이 있는 현재 PC에서는 생략한다. Conda가 설치된 Anaconda Prompt에서 인터넷 연결 후 실행한다.

```bat
conda create -n env_aias_test python=3.12 pip -y
conda activate env_aias_test
python -m pip install jupyterlab ipykernel
cd /d C:\llm_hands_on
jupyter lab
```

이 명령은 Python과 기본 노트북 실행 환경을 준비한다. 교안 폴더는 먼저 받아 두어야 하며, PyTorch 등 수업용 패키지 전체는 실습 요구사항에 맞게 추가 설치해야 한다.

## 참고

- [Jupyter 공식 설치 안내](https://jupyter.org/install): JupyterLab과 Notebook의 설치·실행 명령.
- [기존 VS Code 가이드](PYTHON-SETUP.md)는 앞서 만든 `.venv` 기준이다. 이번에 성공한 Conda 환경을 VS Code에서도 쓰려면 인터프리터와 노트북 커널로 `env_aias_test`를 선택한다.
