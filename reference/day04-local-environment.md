# LLM 서브모듈의 로컬 환경 설정 보관

현재 `llm_hands_on`은 강의 원본 `heury/llm_hands_on`을 연결한 서브모듈이다. 수업 저장소 전체 변경사항을 GitHub에 보관하면서, 강사 원본 저장소에는 변경을 보내지 않도록 로컬 차이를 [패치 파일](day04-llm-local-settings.patch)로 함께 저장했다.

포함된 차이는 두 가지뿐이다.

- `.vscode/settings.json`: 현재 수업 PC의 Python 인터프리터 경로.
- `Chapter_3_Excercise_Attention.ipynb`: 커널 표시 이름을 `env_aias_test`로 변경한 메타데이터. 코드 변경은 없다.

집에서는 Python 설치 위치가 다를 수 있으므로 이 패치를 무조건 적용하지 말고, VS Code/Jupyter에서 집 PC에 설치한 커널을 선택하면 된다. 강의 원본의 코드와 부모 저장소의 서브모듈 커밋 참조는 그대로다. 이 PC의 미커밋 환경 변경도 그대로 유지했다.
