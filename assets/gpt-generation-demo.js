"use strict";
document.querySelectorAll("[data-generation-demo]").forEach(root => {
  const slider = root.querySelector("input[type=range]");
  const sequence = [1, 2, 3, 4, 5, 6, 7];
  const update = () => {
    const step = Number(slider.value);
    const context = sequence.slice(0, 2 + step);
    root.querySelector("[data-step]").textContent = String(step);
    root.querySelector("[data-full]").textContent = JSON.stringify(context);
    root.querySelector("[data-window]").textContent = JSON.stringify(context.slice(-3));
    root.querySelector("[data-next]").textContent = String(sequence[2 + step]);
  };
  slider.addEventListener("input", update);
  update();
});
