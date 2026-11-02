"use strict";
// One query, two keys, two values. A numerical illustration, not a language model.
document.querySelectorAll("[data-attention-demo]").forEach(root => {
  const preference = root.querySelector("[data-query-input]");
  const valueInput = root.querySelector("[data-value-input]");
  const set = (name, text) => {root.querySelector(`[data-${name}]`).textContent = text;};
  const update = () => {
    const t = Number(preference.value) / 10;
    const v = Number(valueInput.value);
    const raw = [2 * t, 2 * (1 - t)];
    const scaled = raw.map(x => x / Math.sqrt(2));
    const peak = Math.max(...scaled);
    const exp = scaled.map(x => Math.exp(x - peak));
    const weights = exp.map(x => x / (exp[0] + exp[1]));
    set("query", `[${t.toFixed(1)}, ${(1-t).toFixed(1)}]`);
    set("value", `[${v}, 0]`);
    set("raw", raw.map(x => x.toFixed(3)).join(", "));
    set("scaled", scaled.map(x => x.toFixed(3)).join(", "));
    set("weights", weights.map(x => x.toFixed(3)).join(", "));
    set("output", `[${(weights[0]*v).toFixed(3)}, ${(weights[1]*10).toFixed(3)}]`);
    set("sum", (weights[0]+weights[1]).toFixed(3));
    const meters = root.querySelectorAll("meter");
    meters.forEach((meter, i) => {meter.value = weights[i];});
  };
  preference.addEventListener("input", update);
  valueInput.addEventListener("input", update);
  update();
});
