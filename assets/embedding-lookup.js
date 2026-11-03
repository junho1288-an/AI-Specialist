"use strict";
document.querySelectorAll("[data-lookup-demo]").forEach(root => {
  const weight = JSON.parse(root.dataset.weight);
  const choice = root.querySelector("select");
  const chips = root.querySelector("[data-input-chips]");
  const output = root.querySelector("[data-output-rows]");
  let ids = [2,3,5,1];
  let selected = 0;
  const format = value => value.toFixed(4);
  const render = () => {
    chips.replaceChildren(); output.replaceChildren();
    ids.forEach((id, index) => {
      const button = document.createElement("button");
      button.type = "button"; button.textContent = String(id);
      button.classList.toggle("is-selected", index === selected);
      button.setAttribute("aria-pressed", String(index === selected));
      button.setAttribute("aria-label", `입력 ${index}번 위치의 ID ${id} 조회`);
      button.addEventListener("click", () => {selected = index; render();});
      chips.appendChild(button);
      const row = document.createElement("tr");
      row.classList.toggle("is-selected", index === selected);
      [`입력 위치 ${index}`,`ID ${id}`, ...weight[id].map(format)].forEach(text => {
        const cell = document.createElement("td"); cell.textContent = text; row.appendChild(cell);
      });
      output.appendChild(row);
    });
    root.querySelectorAll("[data-weight-row]").forEach(row => {
      row.classList.toggle("is-selected", Number(row.dataset.weightRow) === ids[selected]);
    });
    root.querySelector("[data-lookup-note]").textContent = `입력 위치 ${selected}의 값 ${ids[selected]} → weight[${ids[selected]}] 행을 꺼내 출력 ${selected}번째 행에 놓습니다.`;
    root.querySelector("[data-output-shape]").textContent = `(${ids.length}, 3)`;
  };
  choice.addEventListener("change", () => {ids = JSON.parse(choice.value); selected = 0; render();});
  render();
});
