// Run against the local inspector and an isolated Chromium instance with CDP enabled.
const assert = require("node:assert/strict");
const fs = require("node:fs/promises");
const path = require("node:path");

const inspector = "http://127.0.0.1:" + (process.argv[2] || 8765);
const completedRecordingJob = process.argv[3];
const browser = "http://127.0.0.1:9331";
const output = path.resolve(".cache/vis-001-evidence");
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));

async function waitFor(operation, description, timeout = 30_000) {
  const until = Date.now() + timeout;
  do {
    const result = await operation();
    if (result) return result;
    await pause(250);
  } while (Date.now() < until);
  throw new Error("Timed out waiting for " + description);
}

async function main() {
  const page = await fetch(browser + "/json/new?about:blank", {method: "PUT"}).then(response => response.json());
  const connection = new WebSocket(page.webSocketDebuggerUrl);
  await new Promise((resolve, reject) => {
    connection.addEventListener("open", resolve, {once: true});
    connection.addEventListener("error", reject, {once: true});
  });
  const requests = new Map();
  const exceptions = [];
  let sequence = 0;
  connection.addEventListener("message", event => {
    const reply = JSON.parse(event.data);
    if (reply.method === "Runtime.exceptionThrown") exceptions.push(reply.params.exceptionDetails);
    if (!requests.has(reply.id)) return;
    const {resolve, reject} = requests.get(reply.id);
    requests.delete(reply.id);
    if (reply.error) reject(new Error(reply.error.message));
    else resolve(reply.result);
  });
  const send = (method, params = {}) => new Promise((resolve, reject) => {
    const id = ++sequence;
    requests.set(id, {resolve, reject});
    connection.send(JSON.stringify({id, method, params}));
  });
  async function js(fn, ...args) {
    const {result, exceptionDetails} = await send("Runtime.evaluate", {
      expression: `(${fn})(...${JSON.stringify(args)})`, returnByValue: true, awaitPromise: true
    });
    if (exceptionDetails) throw new Error(result.description || exceptionDetails.text);
    return result.value;
  }
  async function screenshot(filename) {
    const {data} = await send("Page.captureScreenshot", {format: "png"});
    await fs.writeFile(path.join(output, filename), Buffer.from(data, "base64"));
  }
  const click = (selector) => js(selector => document.querySelector(selector).click(), selector);
  const select = (selector, value) => js((selector, value) => {
    const node = document.querySelector(selector);
    node.value = value;
    node.dispatchEvent(new Event("change"));
  }, selector, value);
  const plots = () => js(() => [...document.querySelectorAll("#matrix-result:not([hidden]) canvas")].filter(canvas => {
    const {data} = canvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height);
    return data.some((value, index) => index % 4 === 3 && value > 0);
  }).length);
  const jobs = () => fetch(inspector + "/api/jobs").then(response => response.json()).then(data => data.jobs);
  const jobStatus = id => jobs().then(items => items.find(job => job.id === id));
  await fs.mkdir(output, {recursive: true});
  try {
    await send("Runtime.enable");
    await send("Page.enable");
    await send("Emulation.setDeviceMetricsOverride", {width: 1440, height: 1050, deviceScaleFactor: 1, mobile: false});
    await send("Page.navigate", {url: inspector});
    await waitFor(() => js(() => document.querySelector("#version-a").options.length >= 2), "model registry");
    const versions = await fetch(inspector + "/api/versions").then(response => response.json()).then(data => data.versions);
    const a = versions.find(item => item.name === "Qwen3-4B");
    const b = versions.find(item => item.name === "als-r256-mlp");
    assert(a && b);
    await select("#version-a", a.id);
    await select("#version-b", b.id);
    await click("#swap");
    assert.deepEqual(await js(() => ["a", "b"].map(side => document.querySelector("#version-" + side).value)),
                     [b.id, a.id]);
    await click("#swap");
    await waitFor(() => js(() => document.querySelector("#benchmark-deltas").textContent.includes("-80 pp")),
                  "matched development deltas");
    const projection = "model.layers.0.mlp.down_proj";
    const selector = `button[data-module="${projection}"]`;
    await click('[data-tab="architecture"]');
    await js((selector) => {
      const input = document.querySelector("#module-search");
      input.value = "layers.0";
      input.dispatchEvent(new Event("input"));
      document.querySelector(selector).click();
    }, selector);
    const dense = await js(() => document.querySelector("#module-detail").textContent);
    assert.match(dense, /2560 × 9728/);
    await click(".tensor-list button");
    await waitFor(async () => await plots() === 4, "four rendered dense matrix plots");
    await select("#inspect-side", "b");
    await click(selector);
    assert.match(await js(() => document.querySelector("#module-graph").textContent), /256 × 9728.*2560 × 256/);
    await click("#zoom-in");
    assert.equal(await js(() => document.querySelector("#zoom-reset").textContent), "115%");
    const point = await js(() => {
      for (let step = 0; step < 7; step++) document.querySelector("#zoom-in").click();
      const graph = document.querySelector("#graph-viewport");
      graph.scrollIntoView({block: "center"});
      const rect = graph.getBoundingClientRect();
      return {x: rect.left + 260, y: rect.top + 40};
    });
    await send("Input.dispatchMouseEvent", {type: "mousePressed", ...point, button: "left", buttons: 1, clickCount: 1});
    await send("Input.dispatchMouseEvent", {type: "mouseMoved", x: point.x - 130, y: point.y, buttons: 1});
    await send("Input.dispatchMouseEvent", {type: "mouseReleased", x: point.x - 130, y: point.y, button: "left", buttons: 0});
    assert(await js(() => document.querySelector("#graph-viewport").scrollLeft > 0));
    await click("#zoom-reset");
    await js(() => [...document.querySelectorAll("#module-detail button")]
      .find(node => node.textContent === "Inspect logical A × B").click());
    await waitFor(async () => await plots() === 4, "four rendered factorized logical weight plots");
    await js(() => {
      document.querySelector(".matrix-options input[type=checkbox]").checked = true;
      document.querySelector(".matrix-options").requestSubmit();
    });
    await waitFor(async () => await plots() === 5, "singular-value slice plot");
    await js(() => window.scrollTo(0, 95));
    await screenshot("architecture-and-matrix.png");
    await js(() => document.querySelector("#matrix-result .chart-grid").scrollIntoView());
    await screenshot("matrix-plots.png");
    const beforeError = new Set((await jobs()).map(job => job.id));
    await js(() => {
      document.querySelector(".matrix-options input[type=number]").value = "99999";
      document.querySelector(".matrix-options").requestSubmit();
    });
    const failure = await waitFor(async () => (await jobs()).find(job => !beforeError.has(job.id)
      && job.kind === "matrix" && job.status === "failed"), "failed slice shown as a job");
    await click('[data-tab="jobs"]');
    await waitFor(() => js(id => document.querySelector(`[data-job="${id}"]`)?.textContent.includes("outside tensor"),
                           failure.id), "visible recoverable error");
    await click('[data-tab="architecture"]');
    await js(() => [...document.querySelectorAll("#module-detail button")]
      .find(node => node.textContent === "Compare A/B logical weights").click());
    await waitFor(() => js(() => document.querySelector("#matrix-result").textContent.includes("75.66%")),
                  "full logical weight comparison and metrics");
    let done;
    let cancelledJob;
    if (completedRecordingJob) {
      done = await jobStatus(completedRecordingJob);
      cancelledJob = (await jobs()).find(job => job.status === "cancelled")?.id;
    } else {
      const beforeRecord = new Set((await jobs()).map(job => job.id));
      await click('[data-tab="recording"]');
      await click("#all-blocks");
      await js(() => {
        document.querySelector("#capture-modules").value += "\nmodel.layers.0.mlp.down_proj\nmodel.layers.17.mlp.down_proj\nmodel.layers.35.mlp.down_proj";
        document.querySelector("#prompt").value = "The capital of France is";
        document.querySelector("#positions").value = "4, 5";
        document.querySelector("#decode-tokens").value = "1";
        document.querySelector("#verify-parity").checked = true;
        document.querySelector("#record-form").requestSubmit();
      });
      const recordingJob = await waitFor(async () => (await jobs()).find(job => !beforeRecord.has(job.id)
        && job.kind === "record"), "browser-submitted recording");
      await waitFor(() => js(id => document.querySelector(`[data-job="${id}"] progress`) !== null, recordingJob.id),
                    "visible progress");
      await screenshot("background-progress.png");
      const queued = await fetch(inspector + "/api/jobs", {method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify({kind: "matrix", version: a.id, tensor: projection + ".weight", row_start: 28})})
        .then(response => response.json());
      await waitFor(() => js(id => document.querySelector(`[data-job="${id}"] button`)?.textContent === "Cancel", queued.id),
                    "queued cancel control");
      await click(`[data-job="${queued.id}"] button`);
      done = await waitFor(async () => {
        const item = await jobStatus(recordingJob.id);
        return ["complete", "failed", "cancelled"].includes(item.status) && item;
      }, "real original and factorized recording", 460_000);
      await waitFor(async () => (await jobStatus(queued.id)).status === "cancelled", "queued cancellation");
      cancelledJob = queued.id;
    }
    assert.equal(done.status, "complete", done.error);
    await click('[data-tab="jobs"]');
    await waitFor(() => js(id => document.querySelector(`[data-job="${id}"] button`)?.textContent === "Open result",
                           done.id), "recording result link");
    await click(`[data-job="${done.id}"] button`);
    const recordingId = await waitFor(() => js(() => document.querySelector("#comparison.active #recording-select")?.value),
                                      "saved recording opened");
    const recording = await fetch(inspector + "/api/recordings/" + recordingId).then(response => response.json());
    assert(recording.versions.every(item => item.capture_parity.exact));
    assert.equal(recording.comparisons.filter(item => /^model\.layers\.\d+$/.test(item.name)).length, 36);
    assert(recording.local_comparisons.every(item => item.status === "matched"));
    assert(recording.tokens.some(token => token.phase === "decode"));
    await click('.token[data-position="5"]');
    await waitFor(() => js(() => document.querySelector("#comparison.active #comparison-result")?.textContent.includes("Position 5 · decode")),
                  "token diagnostic drill down");
    await js(() => window.scrollTo(0, document.querySelector("#comparison").offsetTop));
    await screenshot("recorded-comparison.png");
    await send("Page.reload", {ignoreCache: true});
    await waitFor(() => js(() => document.querySelector("#comparison.active #comparison-result")?.textContent.includes("Position 5 · decode")),
                  "comparison and token selection reopened after reload");
    await js(() => [...document.querySelectorAll("#comparison-result td button")]
      .find(button => button.textContent === "model.layers.6").click());
    await waitFor(() => js(() => document.querySelector("#architecture.active #module-title")?.textContent === "model.layers.6"),
                  "block drill down opens architecture");
    const beforeImport = new Set((await jobs()).map(job => job.id));
    await click("#add-version");
    await js((name, localPath) => {
      document.querySelector("#new-name").value = name;
      document.querySelector("#new-path").value = localPath;
      document.querySelector("#version-form").requestSubmit();
    }, b.name, b.path);
    const imported = await waitFor(async () => (await jobs()).find(job => !beforeImport.has(job.id)
      && job.kind === "import" && job.status === "complete"), "imported local checkpoint");
    const importResult = await fetch(inspector + "/api/jobs/" + imported.id).then(response => response.json());
    assert.equal(importResult.id, b.id, "re-import keeps existing checkpoint ID");
    assert.deepEqual(exceptions, []);
    await fs.writeFile(path.join(output, "browser-verification.json"), JSON.stringify({
      viewport: [1440, 1050], job: done.id, recording: recordingId, cancelledJob, importJob: imported.id,
      recoveredErrorJob: failure.id, matrixPlots: 5, results: "passed"
    }, null, 2));
    console.log("Desktop UI verified. Recording:", recordingId, "Screenshots:", output);
  } finally {
    connection.close();
    await fetch(browser + "/json/close/" + page.id);
  }
}

main().catch(error => { console.error(error); process.exitCode = 1; });
