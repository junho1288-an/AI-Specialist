"use strict";
document.querySelectorAll("[data-causal-demo]").forEach(root => {
  const slider = root.querySelector("input[type=range]");
  const update = () => {
    const i = Number(slider.value);
    root.querySelector("[data-query-index]").textContent = String(i);
    root.querySelector("[data-visible]").textContent = Array.from({length:i+1}, (_,j)=>j).join(", ");
    root.querySelector("[data-row]").textContent = "[" + Array.from({length:4}, (_,j)=> j<=i ? (1/(i+1)).toFixed(3) : "0.000").join(", ") + "]";
  };
  slider.addEventListener("input", update);
  update();
});
