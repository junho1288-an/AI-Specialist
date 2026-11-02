"use strict";
// A toy token-list slicer. Does not tokenize text or run Python.
document.querySelectorAll("[data-token-window]").forEach(root => {
  const tokens = JSON.parse(root.dataset.tokens);
  const slider = root.querySelector("input[type=range]");
  const size = root.querySelector("[data-size]");
  const x = root.querySelector("[data-x]");
  const y = root.querySelector("[data-y]");
  const update = () => {
    const length = Number(slider.value);
    size.textContent = String(length);
    x.textContent = JSON.stringify(tokens.slice(0, length));
    y.textContent = JSON.stringify(tokens.slice(1, length + 1));
  };
  slider.addEventListener("input", update);
  update();
});
