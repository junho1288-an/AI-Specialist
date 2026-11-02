"use strict";
// Toy visualizations only: no downloads, Python execution, or model training.
document.querySelectorAll("[data-batch-demo]").forEach(root => {
  const sizeInput = root.querySelector("input[type=range]");
  const drop = root.querySelector("input[type=checkbox]");
  const update = () => {
    const size = Number(sizeInput.value);
    const count = 5;
    const batches = [];
    for (let i = 0; i < count; i += size) {
      const ids = Array.from({length: Math.min(size, count - i)}, (_, j) => i + j);
      if (drop.checked && ids.length < size) continue;
      batches.push(ids);
    }
    root.querySelector("[data-batch-size]").textContent = String(size);
    root.querySelector("[data-batch-list]").textContent = JSON.stringify(batches);
    root.querySelector("[data-batch-count]").textContent = String(batches.length);
  };
  sizeInput.addEventListener("input", update);
  drop.addEventListener("change", update);
  update();
});
document.querySelectorAll("[data-embedding-demo]").forEach(root => {
  const table = JSON.parse(root.dataset.table);
  const input = root.querySelector("input[type=range]");
  const update = () => {
    const id = Number(input.value);
    root.querySelector("[data-token-id]").textContent = String(id);
    root.querySelector("[data-vector]").textContent = JSON.stringify(table[id]);
  };
  input.addEventListener("input", update);
  update();
});
