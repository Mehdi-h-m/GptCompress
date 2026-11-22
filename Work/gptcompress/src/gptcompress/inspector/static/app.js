"use strict";

const $ = id => document.getElementById(id);
const state = {versions: [], selected: null, zoom: 1, pending: new Map(), polling: false};
const active = new Set(["queued", "running", "cancelling"]);

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = String(text);
  if (className) node.className = className;
  return node;
}

function button(label, action) {
  const node = element("button", label);
  node.type = "button";
  node.onclick = guarded(action);
  return node;
}

function showError(message) {
  $("error").textContent = message;
  $("error").hidden = !message;
}

async function api(path, body) {
  const options = body === undefined ? {} : {
    method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)
  };
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || response.statusText);
  return data;
}

function guarded(action) {
  return async event => {
    try {
      showError("");
      await action(event);
    } catch (error) {
      showError(error.message);
    }
  };
}

function number(value, digits = 4) {
  return Number.isFinite(value) ? Number(value.toPrecision(digits)).toLocaleString(undefined, {
    maximumSignificantDigits: digits
  }) : "Unavailable";
}

const integer = value => Number.isFinite(value) ? value.toLocaleString() : "Unavailable";
const percent = value => Number.isFinite(value) ? number(100 * value) + "%" : "Unavailable";

function bytes(value) {
  if (!Number.isFinite(value)) return "Unavailable";
  if (value < 1024) return integer(value) + " B";
  const index = Math.min(3, Math.floor(Math.log2(value) / 10));
  return (value / 1024 ** index).toFixed(2) + " " + ["B", "KiB", "MiB", "GiB"][index];
}

function metric(label, value) {
  const node = element("div", undefined, "metric");
  node.append(element("small", label), element("strong", value));
  return node;
}

function details(label, content) {
  const node = element("details");
  node.append(element("summary", label), content);
  return node;
}

function jsonView(value) {
  return element("pre", JSON.stringify(value, null, 2), "detail-json");
}

function table(labels, rows) {
  const node = element("table");
  const header = node.createTHead().insertRow();
  labels.forEach(label => header.append(element("th", label)));
  const body = node.createTBody();
  rows.forEach(values => {
    const row = body.insertRow();
    values.forEach(value => {
      const cell = row.insertCell();
      if (value instanceof Node) cell.append(value);
      else cell.textContent = value;
    });
  });
  return node;
}

const byId = id => state.versions.find(version => version.id === id || version.legacy_ids?.includes(id));
const version = side => byId($("version-" + side).value);
const inspected = () => version($("inspect-side").value);

function tab(name) {
  document.querySelectorAll(".tab").forEach(node => node.classList.toggle("active", node.id === name));
  document.querySelectorAll("nav button").forEach(node => node.classList.toggle("active", node.dataset.tab === name));
  localStorage.setItem("inspector-tab", name);
}
document.querySelectorAll("nav button").forEach(node => { node.onclick = () => tab(node.dataset.tab); });

async function loadVersions() {
  state.versions = (await api("/api/versions")).versions;
  ["a", "b"].forEach((side, index) => {
    const select = $("version-" + side);
    const saved = byId(select.value || localStorage.getItem("inspector-version-" + side));
    select.replaceChildren(...state.versions.map(item => new Option(item.name, item.id)));
    select.value = (saved || state.versions[index] || state.versions[0])?.id || "";
  });
  await renderVersions();
}

async function renderVersions() {
  const a = version("a"), b = version("b");
  $("overview").replaceChildren();
  ["a", "b"].forEach(side => {
    const item = version(side);
    if (!item) return;
    localStorage.setItem("inspector-version-" + side, item.id);
    const card = element("article", undefined, "version-card");
    const values = element("div", undefined, "metrics");
    values.append(metric("Checkpoint storage", bytes(item.bytes)),
                  metric("Stored parameters", integer(item.parameters)),
                  metric("Architecture", item.family));
    card.append(element("h3", side.toUpperCase() + " · " + item.name), values,
                element("p", item.representation, "muted"));
    if (item.parent) card.append(element("p", "Parent · " + (byId(item.parent)?.name || item.parent), "muted"));
    if (Object.keys(item.shared_parameters || {}).length) {
      card.append(element("p", "Tied embedding and LM head share one stored weight.", "muted"));
    }
    card.append(details("Version details", jsonView({
      path: item.path, checkpoint_sha256: item.identity, revision: item.revision,
      tokenizer_sha256: item.tokenizer_identity, transformation: item.transformation,
      capabilities: item.capabilities, unavailable: item.unsupported
    })));
    $("overview").append(card);
  });
  $("all-blocks").disabled = !a?.capabilities.includes("capture");
  $("record-form").querySelector("button[type=submit]").disabled = !a?.capabilities.includes("capture");
  renderTree();
  renderBenchmarkHistory(a, b);
  $("benchmark-deltas").replaceChildren();
  if (!a || !b) return;
  const {deltas} = await api("/api/benchmarks?" + new URLSearchParams({a: a.id, b: b.id}));
  if (version("a")?.id !== a.id || version("b")?.id !== b.id) return;
  $("benchmark-deltas").append(deltas.length ? table(
    ["Benchmark", "Profile", "Samples each", "Version A", "Version B", "B − A"],
    deltas.map(row => [row.task, row.profile, integer(row.n), percent(row.a), percent(row.b),
                       (row.delta_pp > 0 ? "+" : "") + number(row.delta_pp) + " pp"])
  ) : element("p", "No completed scores share a protocol and question IDs.", "muted"));
}

function renderBenchmarkHistory(a, b) {
  const destination = $("benchmarks");
  destination.replaceChildren();
  [a, b].forEach((item, index) => {
    const panel = element("section", undefined, "panel");
    panel.append(element("h3", (index === 0 ? "A" : "B") + " · " + (item?.name || "Unavailable")));
    (item?.experiments || []).forEach(run => {
      const content = element("div");
      content.append(element("p", [
        run.device, run.status, "Elapsed " + (Number.isFinite(run.seconds) ? number(run.seconds) + " s" : "unavailable"),
        "Peak RAM " + bytes(run.resources.peak_rss_bytes), "Peak VRAM " + bytes(run.resources.peak_vram_bytes)
      ].filter(Boolean).join(" · "), "muted"));
      content.append(table(["Benchmark", "Score", "Samples", "Status"], Object.entries(run.tasks).map(
        ([name, values]) => [name, percent(values.score), integer(values.n), values.status || "Unavailable"]
      )));
      content.append(element("p", "Run " + run.id, "identity"));
      panel.append(details((run.profile || "Run") + " · " + run.id, content));
    });
    if (!item?.experiments?.length) panel.append(element("p", "No benchmark records available.", "muted"));
    destination.append(panel);
  });
}

["a", "b"].forEach(side => { $("version-" + side).onchange = guarded(renderVersions); });
$("swap").onclick = guarded(() => {
  const a = $("version-a");
  const b = $("version-b");
  [a.value, b.value] = [b.value, a.value];
  return renderVersions();
});
$("inspect-side").onchange = renderTree;
$("module-search").oninput = renderTree;

function renderTree() {
  const container = $("module-tree");
  container.replaceChildren();
  const query = $("module-search").value.trim().toLowerCase();
  const root = {children: new Map()};
  for (const item of inspected()?.modules || []) {
    if (!item.name.toLowerCase().includes(query)) continue;
    let branch = root;
    for (const segment of item.name.split(".")) {
      if (!branch.children.has(segment)) branch.children.set(segment, {children: new Map()});
      branch = branch.children.get(segment);
    }
    branch.module = item;
  }
  function draw(branch, parent, depth) {
    for (const [name, child] of branch.children) {
      let place = parent;
      if (child.children.size) {
        place = details(child.module?.kind === "transformer block" ? "Block " + name : name,
                        element("div", undefined, "tree-branch"));
        place.open = Boolean(query) || depth < 2;
        parent.append(place);
        place = place.lastElementChild;
      }
      if (child.module) {
        const label = child.children.size ? "Inspect " + child.module.kind : name;
        const select = button(label, () => selectModule(child.module));
        select.title = child.module.name;
        select.dataset.module = child.module.name;
        place.append(select);
      }
      if (child.children.size) draw(child, place, depth + 1);
    }
  }
  draw(root, container, 0);
  if (!root.children.size) container.append(element("p", "No modules match this selection.", "muted"));
  clearModule();
}

function clearModule() {
  state.selected = null;
  $("module-title").textContent = "Architecture explorer";
  $("breadcrumbs").textContent = "";
  $("graph-caption").textContent = "Expand a transformer block or choose a weight to inspect.";
  $("module-detail").replaceChildren();
  $("module-graph").replaceChildren();
  $("matrix-result").hidden = true;
}

function graphNode(label, subtitle, isFactor = false) {
  const node = element("div", label, isFactor ? "node factor" : "node");
  if (subtitle) node.append(element("small", subtitle));
  return node;
}

function selectModule(item) {
  state.selected = item;
  const current = inspected();
  $("breadcrumbs").textContent = current.name + " / " + item.name.split(".").join(" / ");
  $("module-title").textContent = item.name;
  const graph = $("module-graph");
  graph.replaceChildren();
  $("module-tree").querySelectorAll("button[data-module]").forEach(node => {
    node.classList.toggle("selected", node.dataset.module === item.name);
    if (node.dataset.module === item.name) {
      for (let parent = node.parentElement; parent && parent.id !== "module-tree"; parent = parent.parentElement) {
        if (parent.tagName === "DETAILS") parent.open = true;
      }
    }
  });
  const children = current.modules.filter(module =>
    module.name.startsWith(item.name + ".") && !module.name.slice(item.name.length + 1).includes("."));
  if (item.kind === "factorized") {
    $("graph-caption").textContent = "Factorized projection. The implemented forward pass applies B, then A.";
    const [outWidth, inWidth] = item.shape;
    graph.append(graphNode("Input x", integer(inWidth) + " features"), element("span", "→", "arrow"),
                 graphNode("B · Input projection", item.rank + " × " + inWidth, true),
                 element("span", "→", "arrow"), graphNode("A · Output projection", outWidth + " × " + item.rank, true),
                 element("span", "→", "arrow"), graphNode("Output", integer(outWidth) + " features"));
  } else {
    $("graph-caption").textContent = children.length ? "Immediate children in the module hierarchy." : "Selected module.";
    graph.append(graphNode(item.name.split(".").at(-1),
                           item.shape?.join(" × ") || item.kind));
    for (const child of children) graph.append(button(child.name.split(".").at(-1), () => selectModule(child)));
  }
  updateZoom();
  const target = $("module-detail");
  target.replaceChildren();
  const values = element("div", undefined, "metrics section-space");
  values.append(metric("Parameters stored directly", integer(item.parameters)),
                metric("Tensor payload stored directly", bytes(item.bytes)));
  if (item.shape) values.append(metric("Logical weight", item.shape.join(" × ")));
  if (item.output_width) values.append(metric("Output width", integer(item.output_width)));
  target.append(values);
  const actions = element("div", undefined, "actions");
  if (current.capture_locations[item.name]) {
    actions.append(button("Add to recording", event => {
      const names = new Set($("capture-modules").value.split("\n").filter(Boolean));
      names.add(item.name);
      $("capture-modules").value = [...names].join("\n");
      event.currentTarget.textContent = "Added to recording";
    }));
  }
  if (item.kind === "factorized") {
    actions.append(button("Inspect logical A × B", () => requestMatrix({module: item.name})));
  }
  if (item.matrix && version("a") && version("b")) {
    actions.append(button("Compare A/B logical weights", () => requestWeightComparison(item.name)));
  }
  target.append(actions);
  const tensors = element("div", undefined, "tensor-list");
  for (const tensor of item.tensors) {
    const inspect = button(tensor.name, () => requestMatrix({tensor: tensor.name}));
    inspect.append(element("small", tensor.shape.join(" × ") + " · " + tensor.dtype + " · " + bytes(tensor.bytes)));
    tensors.append(inspect);
  }
  target.append(tensors);
  $("matrix-result").hidden = true;
}

function updateZoom(amount = state.zoom) {
  state.zoom = Math.max(.4, Math.min(2, amount));
  $("module-graph").style.zoom = state.zoom;
  $("zoom-reset").textContent = Math.round(100 * state.zoom) + "%";
}
$("zoom-in").onclick = () => updateZoom(state.zoom + .15);
$("zoom-out").onclick = () => updateZoom(state.zoom - .15);
$("zoom-reset").onclick = () => updateZoom(1);

const viewport = $("graph-viewport");
let drag;
viewport.onpointerdown = event => {
  if (event.target.closest("button")) return;
  drag = {x: event.clientX, y: event.clientY, left: viewport.scrollLeft, top: viewport.scrollTop};
  viewport.setPointerCapture(event.pointerId);
};
viewport.onpointermove = event => {
  if (!drag) return;
  viewport.scrollLeft = drag.left - event.clientX + drag.x;
  viewport.scrollTop = drag.top - event.clientY + drag.y;
};
viewport.onlostpointercapture = () => { drag = null; };

async function requestMatrix(selection, options = {}, versionId = inspected().id) {
  const context = {selection, versionId, selected: state.selected?.name};
  const response = await api("/api/jobs", {kind: "matrix", version: versionId, ...selection, ...options});
  state.pending.set(response.id, context);
  pendingMatrix("Computing matrix slice");
  await poll();
}

async function requestWeightComparison(module) {
  const response = await api("/api/jobs", {kind: "matrix_compare", module, a: version("a").id, b: version("b").id});
  state.pending.set(response.id, {selected: module, versionId: inspected().id});
  pendingMatrix("Comparing full logical weights");
  await poll();
}

function pendingMatrix(message) {
  $("matrix-result").hidden = false;
  $("matrix-result").replaceChildren(element("p", message + ". Track progress in Jobs.", "muted"));
}

function chart(title, draw) {
  const box = element("div", undefined, "chart-box");
  const canvas = element("canvas");
  canvas.width = 640;
  canvas.height = 260;
  canvas.setAttribute("role", "img");
  canvas.setAttribute("aria-label", title);
  box.append(element("h3", title), canvas);
  draw(canvas);
  return box;
}

function plot(canvas, values, bars = false, range = null, format = number) {
  const context = canvas.getContext("2d");
  const valid = values.filter(Number.isFinite);
  context.font = "14px Segoe UI";
  context.fillStyle = "#a2b4c7";
  if (!valid.length) {
    context.fillText("Unavailable", 20, 35);
    return;
  }
  const maximum = Math.max(...valid, 1e-12);
  context.fillText(format(maximum), 8, 18);
  context.fillText(range ? number(range[0]) : format(0), 36, 253);
  if (range) {
    context.textAlign = "right";
    context.fillText(number(range.at(-1)), 612, 253);
    context.textAlign = "left";
  }
  const spacing = 576 / Math.max(values.length, 1);
  const curve = (value, i) => [36 + (i + .5) * spacing, 232 - value / maximum * 204];
  context.strokeStyle = "#283b49";
  context.beginPath();
  context.moveTo(36, 232);
  context.lineTo(612, 232);
  context.stroke();
  context.fillStyle = context.strokeStyle = "#68d9bd";
  context.lineWidth = 2;
  let drawing = false;
  context.beginPath();
  values.forEach((value, i) => {
    if (!Number.isFinite(value)) { drawing = false; return; }
    const [x, y] = curve(value, i);
    if (bars) context.fillRect(x - spacing * .42, y, spacing * .84, 232 - y);
    else if (drawing) context.lineTo(x, y);
    else { context.moveTo(x, y); drawing = true; }
  });
  if (!bars) context.stroke();
}

function heatmap(canvas, result) {
  const values = result.heatmap;
  const [rows, columns] = [values.length, values[0].length];
  const max = Math.max(1e-12, ...values.flat().filter(Number.isFinite).map(Math.abs));
  const context = canvas.getContext("2d");
  const [width, height] = [canvas.width / columns, canvas.height / rows];
  values.forEach((line, y) => line.forEach((value, x) => {
    const t = Math.abs(value) / max;
    context.fillStyle = !Number.isFinite(value) ? "#a258a0" : value >= 0
      ? `rgb(${21 + t * 76},${34 + t * 185},${43 + t * 136})`
      : `rgb(${21 + t * 229},${34 + t * 146},${43 + t * 75})`;
    context.fillRect(x * width, y * height, Math.ceil(width), Math.ceil(height));
  }));
  canvas.title = "Color scale ±" + number(max);
  canvas.onmousemove = event => {
    const rect = canvas.getBoundingClientRect();
    const x = Math.min(columns - 1, Math.floor((event.clientX - rect.left) / rect.width * columns));
    const y = Math.min(rows - 1, Math.floor((event.clientY - rect.top) / rect.height * rows));
    canvas.title = `Row ${result.row_start + y * result.heatmap_step}, column ${result.col_start + x * result.heatmap_step} · ${number(values[y][x])}`;
  };
}

function renderMatrix(result, context) {
  const target = $("matrix-result");
  target.hidden = false;
  target.replaceChildren(element("h3", result.tensor));
  target.append(element("p", result.sampling, "muted"));
  const values = element("div", undefined, "stats-grid");
  const labels = {min: "Minimum", max: "Maximum", mean: "Mean", std: "Std. deviation", norm: "Frobenius norm", nonfinite: "Nonfinite cells"};
  Object.entries(result.stats).forEach(([key, value]) => values.append(metric(labels[key], number(value))));
  target.append(values);
  const grid = element("div", undefined, "chart-grid");
  grid.append(chart("Weights · teal positive / amber negative", canvas => heatmap(canvas, result)),
              chart("Histogram", canvas => plot(canvas, result.histogram.counts, true, result.histogram.edges)),
              chart("Row norms", canvas => plot(canvas, result.row_norms)),
              chart("Column norms", canvas => plot(canvas, result.column_norms)));
  if (result.spectrum.length) grid.append(chart("Singular values of this slice", canvas => plot(canvas, result.spectrum)));
  target.append(grid);
  if (!context?.selection) return;
  const form = element("form", undefined, "matrix-options");
  const inputs = [result.row_start, result.col_start].map((value, i) => {
    const label = element("label", i === 0 ? "Row offset" : "Column offset");
    const input = element("input");
    input.type = "number";
    input.min = 0;
    input.value = value;
    label.append(input);
    form.append(label);
    return input;
  });
  const spectral = element("input");
  spectral.type = "checkbox";
  spectral.checked = Boolean(result.spectrum.length);
  const label = element("label", "Slice spectrum (SVD)", "check");
  label.prepend(spectral);
  form.append(label, element("button", "Inspect slice"));
  form.onsubmit = guarded(event => {
    event.preventDefault();
    return requestMatrix(context.selection, {row_start: Number(inputs[0].value),
      col_start: Number(inputs[1].value), spectrum: spectral.checked}, context.versionId);
  });
  target.append(form);
}

function renderWeightComparison(result) {
  const target = $("matrix-result");
  target.hidden = false;
  target.replaceChildren(element("h3", result.module + " · A/B weights"));
  if (result.status !== "matched") {
    target.append(element("p", result.status, "notice"));
    return;
  }
  const values = element("div", undefined, "metrics");
  values.append(metric("Relative error vs A", percent(result.relative_error)), metric("Cosine similarity", number(result.cosine)));
  if (result.source_version) {
    values.append(metric("Squared reconstruction error vs dense source", percent(result.relative_squared_reconstruction_error)),
                  metric("Source spectral energy in factor span", percent(result.retained_spectral_energy)));
  }
  target.append(values, element("p", result.basis, "muted"));
  if (result.energy_basis) target.append(element("p", result.source_version + " · " + result.energy_basis, "muted"));
  if (result.preview) {
    target.append(chart("Weight change B − A · sampled from the first " + result.preview.extent.join(" × "),
                        canvas => heatmap(canvas, result.preview)));
  }
}

$("all-blocks").onclick = () => {
  const names = new Set($("capture-modules").value.split("\n").filter(Boolean));
  Object.entries(version("a")?.capture_locations || {}).forEach(([name, kind]) => {
    if (kind === "transformer block") names.add(name);
  });
  $("capture-modules").value = [...names].join("\n");
};

$("record-form").onsubmit = guarded(async event => {
  event.preventDefault();
  const modules = $("capture-modules").value.split("\n").map(value => value.trim()).filter(Boolean);
  const items = $("positions").value.split(",").map(value => value.trim()).filter(Boolean);
  if (items.some(value => !/^\d+$/.test(value))) throw new Error("Choose nonnegative whole token positions.");
  await api("/api/jobs", {kind: "record", a: version("a")?.id, b: $("single-version").checked ? null : version("b")?.id,
    prompt: $("prompt").value, max_tokens: Number($("max-tokens").value), decode_tokens: Number($("decode-tokens").value),
    modules, positions: items.map(Number), raw: $("raw").checked, verify_parity: $("verify-parity").checked});
  tab("jobs");
  await poll();
});

async function loadRecordings() {
  const {recordings} = await api("/api/recordings");
  const list = $("recordings"), select = $("recording-select"), previous = select.value;
  list.replaceChildren();
  select.replaceChildren(new Option("Choose a recording", ""));
  for (const item of recordings) {
    const row = element("div", undefined, "saved-row");
    const open = button(item.prompt, () => openRecording(item.id));
    open.append(element("small", new Date(item.created).toLocaleString() + " · " + item.status));
    row.append(open);
    list.append(row);
    select.add(new Option(item.prompt.slice(0, 55) + " · " + item.status + " · " + item.id, item.id));
  }
  if (!recordings.length) list.append(element("p", "No saved recordings.", "empty"));
  select.value = previous;
}

async function openRecording(id, focus = true) {
  if (!id) return;
  const result = await api("/api/recordings/" + encodeURIComponent(id));
  if (focus) {
    for (const side of ["a", "b"]) {
      const item = byId(result[side]);
      if (item) $("version-" + side).value = item.id;
    }
    await renderVersions();
    tab("comparison");
  }
  $("recording-select").value = id;
  localStorage.setItem("inspector-recording", id);
  renderComparison(result);
}
$("recording-select").onchange = guarded(() => openRecording($("recording-select").value));

async function showRecordedModule(result, module) {
  const a = byId(result.a);
  if (!a) throw new Error("The recorded checkpoint is unavailable in the registry.");
  $("version-a").value = a.id;
  $("inspect-side").value = "a";
  $("module-search").value = "";
  await renderVersions();
  const item = a.modules.find(item => item.name === module);
  if (!item) throw new Error("Module unavailable in the recorded checkpoint.");
  selectModule(item);
  tab("architecture");
}

function renderComparison(result) {
  const target = $("comparison-result");
  target.className = "";
  target.replaceChildren(element("h3", result.prompt));
  const a = byId(result.a)?.name || result.versions[0]?.name || result.a;
  const b = byId(result.b)?.name || result.versions[1]?.name || result.b;
  target.append(element("p", [`A: ${a}`, b && `B: ${b}`, result.status].filter(Boolean).join("  ·  "), "muted"));
  if (result.error) target.append(element("p", result.error, "notice"));
  const tokens = element("div", undefined, "tokens");
  const phases = element("p", undefined, "muted");
  const metrics = element("div");
  const unavailable = result.status !== "complete";
  const data = result.probability_comparisons || [];
  const selectPosition = position => {
    tokens.querySelectorAll("button").forEach(token =>
      token.setAttribute("aria-pressed", String(
        (token.dataset.position === undefined ? null : Number(token.dataset.position)) === position)));
    phases.textContent = position === null ? "Selected positions combined" :
      `Position ${position} · ${result.tokens.find(token => token.position === position)?.phase || "prefill"}`;
    renderActivationMetrics(metrics, result, position);
    localStorage.setItem("inspector-position", JSON.stringify({recording: result.id, position}));
  };
  const all = button("All captured positions", () => selectPosition(null));
  all.setAttribute("aria-pressed", "true");
  tokens.append(all);
  for (const token of result.tokens || []) {
    const pill = button(token.position + " · " + (token.text || "∅"), () => selectPosition(token.position));
    pill.classList.add("token");
    pill.dataset.position = token.position;
    pill.title = [token.phase, "ID " + token.id].filter(Boolean).join(" · ");
    pill.setAttribute("aria-pressed", "false");
    pill.disabled = !result.positions?.includes(token.position) && !data.some(value => value.position === token.position);
    tokens.append(pill);
  }
  target.append(tokens, phases, metrics);
  if (unavailable) target.append(element("p", "This recording contains partial results.", "muted"));
  const savedPosition = JSON.parse(localStorage.getItem("inspector-position") || "null");
  selectPosition(savedPosition?.recording === result.id ? savedPosition.position : null);
  target.append(element("h3", "Recorded resources"));
  target.append(table(["Version", "Load", "Hooked forward", "Total", "Sampled peak RAM", "Logit parity"],
    (result.versions || []).map(item => [item.name || item.id, duration(item.load_seconds), duration(item.forward_seconds),
      duration(item.seconds), bytes(item.peak_rss_bytes), item.capture_parity ? (item.capture_parity.exact ? "Bitwise equal" : "Different") : "Unchecked"])));
  target.append(element("p", "Recording time includes instrumentation. Activation differences identify where outputs diverge and do not establish cause.", "muted"));
  target.append(details("Recording settings", jsonView({settings: result.settings, versions: result.versions.map(
    ({id, name, identity}) => ({id, name, identity})), notes: result.notes})));
}

const duration = seconds => Number.isFinite(seconds) ? number(seconds) + " s" : "Unavailable";

function renderActivationMetrics(target, result, position) {
  target.replaceChildren();
  if (result.comparisons?.length) {
    const entries = result.comparisons.map(row => ({
      ...row, measure: position === null ? row : row.by_position?.find(item => item.position === position)
    }));
    const blocks = entries.filter(row => /^model\.layers\.\d+$/.test(row.name));
    if (blocks.length) target.append(chart("Relative output error by recorded block",
      canvas => plot(canvas, blocks.map(row => row.measure?.relative_error), true,
                     [Number(blocks[0].name.split(".").at(-1)), Number(blocks.at(-1).name.split(".").at(-1))],
                     percent)));
    target.append(table(["Module", "Relative output error", "Cosine similarity", "Alignment"],
      entries.map(row => [button(row.name, () => showRecordedModule(result, row.name)),
        percent(row.measure?.relative_error), number(row.measure?.cosine), row.measure?.status || "Position unavailable"])));
    target.append(element("p", "Whole-model forward passes on identical token IDs. Relative errors use Version A as the reference.", "muted"));
  }
  if (result.local_comparisons?.length) {
    target.append(element("h3", "Local projection replay"));
    target.append(element("p", "Each pair of projections receives the same Version A input. Measurements combine the positions shown.", "muted"));
    target.append(table(["Projection", "Positions", "Relative error", "Cosine similarity", "Status"],
      result.local_comparisons.map(row => [row.name, row.positions?.join(", ") || "Unavailable",
        percent(row.relative_error), number(row.cosine), row.status])));
  }
  const probability = position === null ? result.probability_comparisons?.at(-1) :
    result.probability_comparisons?.find(row => row.position === position);
  if (probability) renderProbabilities(target, probability);
  else if (result.probability_comparison) target.append(details("Saved next-token comparison", jsonView(result.probability_comparison)));
  for (const item of result.versions || []) {
    const version = element("div", undefined, "capture-groups");
    const entries = Object.entries(item.captures || {});
    const rows = entries.flatMap(([name, entry]) => Object.entries(entry.positions || {})
      .filter(([index]) => position === null || Number(index) === position)
      .flatMap(([index, data]) => Object.entries(data).map(([kind, capture]) =>
        [name, index, kind, number(capture.stats.norm), number(capture.stats.min),
         number(capture.stats.max), number(capture.stats.mean), number(capture.stats.std)])));
    version.append(table(["Module", "Position", "Capture", "Norm", "Minimum", "Maximum", "Mean", "SD"], rows));
    version.append(details("Capture details and requested raw slices", jsonView(item.captures || {})));
    target.append(details("Activations · " + (item.name || item.id), version));
    if (!probability) target.append(details("Top next tokens · " + (item.name || item.id), table(
      ["Token", "Probability"], (item.top_tokens || []).map(row => [row.text || row.id, percent(row.probability)]))));
  }
}

function renderProbabilities(target, row) {
  target.append(element("h3", "Next-token distribution after position " + row.position));
  if (row.status !== "matched") {
    target.append(element("p", row.status, "notice"));
    return;
  }
  target.append(metric("Total variation distance", percent(row.total_variation)));
  target.append(details("Largest probability changes", table(["Token", "ID", "Version A", "Version B", "B − A"],
    row.largest_deltas.map(token => [token.text, token.id, percent(token.a_probability), percent(token.probability),
                                    number(token.delta * 100) + " pp"]))));
  for (const side of ["a", "b"]) {
    target.append(details("Top predictions · Version " + side.toUpperCase(), table(["Token", "ID", "Probability"],
      row[side + "_top"].map(token => [token.text, token.id, percent(token.probability)]))));
  }
}

async function openJob(job) {
  const result = await api("/api/jobs/" + job.id);
  const context = state.pending.get(job.id);
  if (job.kind === "record") return openRecording(result.recording_id);
  if (job.kind === "import") return loadVersions();
  if (context?.versionId !== undefined) {
    const id = context.versionId;
    if (version("a")?.id !== id && version("b")?.id !== id && byId(id)) {
      $("version-a").value = id;
      await renderVersions();
    }
    $("inspect-side").value = version("a")?.id === id ? "a" : "b";
    renderTree();
    const module = inspected()?.modules.find(item => item.name === context.selected);
    if (module) selectModule(module);
  }
  if (job.kind === "matrix") renderMatrix(result, context);
  else renderWeightComparison(result);
  tab("architecture");
}

let lastJobs = "";
const completed = new Set();
async function poll() {
  if (state.polling) return;
  state.polling = true;
  try {
    const {jobs} = await api("/api/jobs");
    const stamp = JSON.stringify(jobs);
    if (lastJobs === stamp) return;
    lastJobs = stamp;
    $("job-count").textContent = jobs.filter(job => active.has(job.status)).length || "";
    const list = $("jobs-list");
    list.replaceChildren();
    for (const job of jobs) {
      const entry = element("div", undefined, "job " + job.status);
      entry.dataset.job = job.id;
      const heading = element("div", undefined, "job-heading");
      const kind = {matrix: "Matrix slice", matrix_compare: "Weight comparison", record: "Activation recording", import: "Checkpoint import"};
      heading.append(element("strong", kind[job.kind] + " · " + job.status));
      if (job.status === "queued" || job.status === "running") {
        heading.append(button("Cancel", async () => {
          await api("/api/jobs/" + job.id + "/cancel", {});
          await poll();
        }));
      }
      entry.append(heading, element("p", job.message));
      if (job.error) entry.append(element("p", job.error, "notice"));
      const meter = element("progress");
      meter.max = 1;
      meter.value = job.progress;
      entry.append(meter);
      if (job.status === "complete") entry.append(button("Open result", () => openJob(job)));
      list.append(entry);
      if (completed.has(job.id) || active.has(job.status)) continue;
      completed.add(job.id);
      if (job.kind === "record") await loadRecordings();
      if (job.kind === "import" && job.status === "complete") await loadVersions();
      const context = state.pending.get(job.id);
      if (context && context.versionId === inspected()?.id && context.selected === state.selected?.name) {
        if (job.status === "complete") {
          const result = await api("/api/jobs/" + job.id);
          if (job.kind === "matrix") renderMatrix(result, context);
          else renderWeightComparison(result);
        } else {
          $("matrix-result").replaceChildren(element("p", job.error || job.message, "notice"));
        }
      }
    }
    if (!jobs.length) list.append(element("p", "No jobs submitted.", "empty"));
    const visible = new Set(jobs.map(job => job.id));
    for (const id of state.pending.keys()) if (!visible.has(id)) state.pending.delete(id);
    for (const id of completed) if (!visible.has(id)) completed.delete(id);
  } finally {
    state.polling = false;
  }
}

$("add-version").onclick = () => $("version-dialog").showModal();
$("close-dialog").onclick = () => $("version-dialog").close();
$("version-form").onsubmit = guarded(async event => {
  event.preventDefault();
  await api("/api/versions", {name: $("new-name").value, path: $("new-path").value});
  $("version-dialog").close();
  tab("jobs");
  await poll();
});

async function initialize() {
  await loadVersions();
  await loadRecordings();
  await poll();
  tab(localStorage.getItem("inspector-tab") || "architecture");
  const saved = localStorage.getItem("inspector-recording");
  if (saved && [...$("recording-select").options].some(option => option.value === saved)) {
    await openRecording(saved, false);
  }
  setInterval(guarded(poll), 1500);
}
guarded(initialize)();
