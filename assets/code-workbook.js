"use strict";
// Static recall checks only: never evaluates or executes student code.
(() => {
  const data = window.resnetWorkbook;
  if (!Array.isArray(data)) return;
  const node = (tag, text, className) => {
    const el = document.createElement(tag);
    if (text !== undefined) el.textContent = text;
    if (className) el.className = className;
    return el;
  };
  const norm = s => s.trim().replace(/\s+/g, "").replace(/'/g, '"');
  data.forEach((item, index) => {
    const step = document.getElementById(item.step);
    if (!step) return;
    const card = node("article", undefined, "code-workbook");
    card.append(node("h3", item.title), node("p", item.source, "source"));
    const theory = node("div", undefined, "workbook-theory");
    theory.append(node("h4", "이론을 코드에 연결하기"));
    item.theory.forEach(t => theory.append(node("p", t)));
    card.append(theory);
    card.append(node("p", "입력 → 출력: " + item.shape, "workbook-shape"));
    card.append(node("h4", "먼저 내 말로 설명"), node("p", item.question));
    const reflection = node("textarea");
    reflection.rows = 3;
    reflection.setAttribute("aria-label", item.title + " 이유 설명");
    reflection.placeholder = "숫자·축·이유를 한두 문장으로 적어 보세요.";
    card.append(reflection);
    const toggle = node("button", "위 설명 가리고 풀기");
    toggle.type = "button";
    toggle.setAttribute("aria-pressed", "false");
    toggle.addEventListener("click", () => {
      const active = step.classList.toggle("workbook-recall");
      step.querySelectorAll(".recall-toggle").forEach(b => {
        b.textContent = active ? "위 설명 다시 보기" : "위 설명 가리고 풀기";
        b.setAttribute("aria-pressed", String(active));
      });
    });
    toggle.className = "recall-toggle";
    card.append(toggle, node("h4", "코드 빈칸 문제"), node("p", item.task));
    const pre = node("pre"); pre.append(node("code", item.code)); card.append(pre);
    const form = node("form");
    const fields = item.blanks.map((blank, i) => {
      const label = node("label", `${i + 1}. ${blank.label}`);
      const input = node("input");
      input.type = "text"; input.autocomplete = "off"; input.spellcheck = false;
      input.id = `workbook-${index}-blank-${i}`;
      label.htmlFor = input.id; label.append(input); form.append(label);
      return input;
    });
    const check = node("button", "빈칸 확인"); check.type = "submit";
    const reveal = node("button", "시도 후 정답·해설 보기"); reveal.type = "button"; reveal.disabled = true;
    const reset = node("button", "이 문제 다시 풀기"); reset.type = "button";
    const actions = node("div", undefined, "workbook-actions"); actions.append(check, reveal, reset); form.append(actions);
    const feedback = node("p", undefined, "feedback"); feedback.setAttribute("aria-live", "polite"); form.append(feedback);
    const result = node("div", undefined, "workbook-result"); result.hidden = true;
    result.append(node("h4", "예시 답안 · 내 설명과 비교"));
    const answer = item.code.replace(/\?\?\?\?\[(\d+)\]/g, (_, n) => item.blanks[Number(n) - 1].answers[0]);
    const solved = node("pre"); solved.append(node("code", answer)); result.append(solved);
    result.append(node("p", item.explanation), node("p", "설명 비교 기준: " + item.theoryAnswer));
    result.append(node("p", "이 결과는 예시 답안과의 텍스트 비교입니다. Python 실행·의미상 동등한 모든 답안의 자동 채점은 아닙니다. 다른 풀이가 맞다고 생각되면 코드와 단계 번호를 보내 주세요.", "source"));
    form.addEventListener("submit", e => {
      e.preventDefault();
      if (fields.some(f => !f.value.trim())) {
        feedback.textContent = "빈칸을 모두 시도해 주세요. 모르겠으면 아래 힌트를 먼저 열어 보세요.";
        feedback.className = "feedback"; return;
      }
      reveal.disabled = false;
      const misses = fields.flatMap((f, i) => item.blanks[i].answers.some(a => norm(a) === norm(f.value)) ? [] : [i + 1]);
      feedback.className = "feedback " + (misses.length ? "incorrect" : "correct");
      feedback.textContent = misses.length
        ? `${misses.join(", ")}번을 다시 확인해 보세요. 힌트 또는 시도 후 해설을 이용할 수 있습니다.`
        : "빈칸이 예시 답안과 일치합니다. 위에 적은 이유·shape도 해설과 비교해 주세요. Python을 실행한 결과는 아닙니다.";
    });
    reveal.addEventListener("click", () => { result.hidden = !result.hidden; reveal.textContent = result.hidden ? "정답·해설 보기" : "정답·해설 접기"; });
    reset.addEventListener("click", () => {
      form.reset(); reflection.value = ""; feedback.textContent = ""; feedback.className = "feedback";
      result.hidden = true; reveal.disabled = true; reveal.textContent = "시도 후 정답·해설 보기"; fields[0].focus();
    });
    card.append(form);
    const hint = node("details"); hint.append(node("summary", "막힐 때 개념 힌트"), node("p", item.hint));
    card.append(hint, result);
    const transfer = node("p", "나중에 Jupyter에서: " + item.transfer, "workbook-transfer");
    card.append(transfer);
    step.append(card);
  });
})();
