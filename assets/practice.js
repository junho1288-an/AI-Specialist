"use strict";
// Small reusable checks for static, local HTML lessons. No Python execution.
document.querySelectorAll("form[data-answer]").forEach((form) => {
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const entry = new FormData(form).get("answer");
    const value = entry === null ? "" : String(entry).trim();
    const output = form.querySelector(".feedback");
    if (!value) {
      output.className = "feedback";
      output.textContent = "답을 먼저 입력하거나 선택해 보세요.";
      return;
    }
    const accepted = form.dataset.answer.split("|");
    const correct = accepted.includes(value);
    output.className = "feedback " + (correct ? "correct" : "incorrect");
    output.textContent = (correct ? "정답입니다. " : "다시 생각해 보세요. ") + form.dataset.explanation;
  });
});
const toggle = document.getElementById("exam-toggle");
if (toggle) toggle.addEventListener("click", () => {
  const active = document.body.classList.toggle("exam-mode");
  toggle.setAttribute("aria-pressed", String(active));
  toggle.textContent = active ? "설명 다시 보기" : "설명 가리고 시험 보기";
  if (active) document.getElementById("exam").scrollIntoView({behavior:"smooth"});
});
const reset = document.getElementById("quiz-reset");
if (reset) reset.addEventListener("click", () => {
  document.querySelectorAll(".quiz form").forEach(form => form.reset());
  document.querySelectorAll(".feedback").forEach(output => {
    output.textContent = "";
    output.className = "feedback";
  });
  document.querySelectorAll("#exam details").forEach(detail => {detail.open = false;});
});
