"use strict";
// Visualizes the notebook's range/slice rules using fixed toy token IDs.
document.querySelectorAll("[data-dataset-windows]").forEach(root => {
  const tokens = JSON.parse(root.dataset.tokens);
  const lengthInput = root.querySelector("[data-length-input]");
  const strideInput = root.querySelector("[data-stride-input]");
  const rows = root.querySelector("tbody");
  const update = () => {
    const length = Number(lengthInput.value);
    const stride = Number(strideInput.value);
    root.querySelector("[data-length]").textContent = String(length);
    root.querySelector("[data-stride]").textContent = String(stride);
    rows.replaceChildren();
    const starts = [];
    for (let i = 0; i < tokens.length - length; i += stride) {
      const row = document.createElement("tr");
      [starts.length, i, JSON.stringify(tokens.slice(i, i + length)),
        JSON.stringify(tokens.slice(i + 1, i + length + 1))].forEach(value => {
        const cell = document.createElement("td");
        cell.textContent = String(value);
        row.appendChild(cell);
      });
      rows.appendChild(row);
      starts.push(i);
    }
    root.querySelector("[data-starts]").textContent = JSON.stringify(starts);
    root.querySelector("[data-count]").textContent = String(starts.length);
  };
  lengthInput.addEventListener("input", update);
  strideInput.addEventListener("input", update);
  update();
});
