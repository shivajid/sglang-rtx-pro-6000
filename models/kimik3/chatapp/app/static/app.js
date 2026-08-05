/* K3 console — no framework, no build step. */

const $ = (sel) => document.querySelector(sel);

const thread = $("#thread");
const input = $("#input");
const tray = $("#tray");
const hint = $("#hint");
const sendBtn = $("#send");
const stopBtn = $("#stop");
const inputRow = document.querySelector(".input-row");

const MAX_EDGE = 1568;          // longest side sent upstream
const STORE_KEY = "k3-console-v1";

let cfg = { chat_model: "Kimi-K3", image_enabled: false, max_image_mb: 8, default_temperature: 0.6 };
let messages = [];              // {role, text, images[], reasoning, meta}
let pending = [];               // data URLs staged in the composer
let mode = "chat";
let controller = null;

/* ------------------------------------------------------------ helpers -- */

const esc = (s) => s.replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

/* Small markdown subset: fenced code, inline code, bold, italic, links, lists. */
function render(md) {
  const blocks = [];
  let text = md.replace(/```(\w*)\n?([\s\S]*?)```/g, (_, lang, code) => {
    blocks.push(`<pre><code data-lang="${esc(lang)}">${esc(code.replace(/\n$/, ""))}</code></pre>`);
    return `\u0000${blocks.length - 1}\u0000`;
  });
  text = esc(text)
    .replace(/`([^`\n]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[\s(])\*([^*\n]+)\*/g, "$1<em>$2</em>")
    .replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g,
      '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');

  const out = [];
  let list = null;
  for (const line of text.split("\n")) {
    const bullet = line.match(/^\s*[-*]\s+(.*)/);
    const numbered = line.match(/^\s*(\d+)\.\s+(.*)/);
    const heading = line.match(/^(#{1,4})\s+(.*)/);
    if (bullet || numbered) {
      const want = bullet ? "ul" : "ol";
      if (list !== want) { if (list) out.push(`</${list}>`); out.push(`<${want}>`); list = want; }
      out.push(`<li>${(bullet ? bullet[1] : numbered[2])}</li>`);
      continue;
    }
    if (list) { out.push(`</${list}>`); list = null; }
    if (heading) out.push(`<h3>${heading[2]}</h3>`);
    else if (line.trim() === "") out.push("");
    else out.push(`<p>${line}</p>`);
  }
  if (list) out.push(`</${list}>`);
  return out.join("\n").replace(/\u0000(\d+)\u0000/g, (_, i) => blocks[+i]);
}

const stamp = () => new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

function say(text, bad = false) {
  hint.textContent = text;
  hint.classList.toggle("bad", bad);
}

function autosize() {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 220) + "px";
}

function scrollDown(force = false) {
  const near = thread.scrollHeight - thread.scrollTop - thread.clientHeight < 160;
  if (near || force) thread.scrollTop = thread.scrollHeight;
}

/* ------------------------------------------------------------- storage -- */

function save() {
  const trim = (keepImages) => JSON.stringify(messages.map((m) => ({
    ...m, images: keepImages ? m.images : [], reasoning: (m.reasoning || "").slice(0, 4000),
  })));
  try {
    localStorage.setItem(STORE_KEY, trim(true));
  } catch {
    try { localStorage.setItem(STORE_KEY, trim(false)); } catch { /* history is optional */ }
  }
}

function restore() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORE_KEY) || "[]");
    if (Array.isArray(saved) && saved.length) {
      messages = saved;
      messages.forEach(paint);
      $("#empty")?.remove();
      scrollDown(true);
    }
  } catch { /* start clean */ }
}

/* -------------------------------------------------------------- images -- */

async function shrink(file) {
  const limit = cfg.max_image_mb * 1024 * 1024;
  if (file.size > limit * 4) throw new Error(`${file.name} is too large to process.`);
  const bitmap = await createImageBitmap(file);
  const scale = Math.min(1, MAX_EDGE / Math.max(bitmap.width, bitmap.height));
  const w = Math.round(bitmap.width * scale);
  const h = Math.round(bitmap.height * scale);
  const canvas = document.createElement("canvas");
  canvas.width = w; canvas.height = h;
  canvas.getContext("2d").drawImage(bitmap, 0, 0, w, h);
  bitmap.close?.();
  let quality = 0.88;
  let url = canvas.toDataURL("image/jpeg", quality);
  while (url.length * 0.75 > limit && quality > 0.4) {
    quality -= 0.12;
    url = canvas.toDataURL("image/jpeg", quality);
  }
  return url;
}

async function stage(files) {
  for (const file of files) {
    if (!file.type.startsWith("image/")) continue;
    try {
      pending.push(await shrink(file));
    } catch (err) {
      say(err.message, true);
    }
  }
  paintTray();
}

function paintTray() {
  tray.innerHTML = "";
  tray.hidden = pending.length === 0;
  pending.forEach((url, i) => {
    const chip = document.createElement("div");
    chip.className = "chip";
    chip.innerHTML = `<img src="${url}" alt="Attachment ${i + 1}"><button type="button" aria-label="Remove attachment">×</button>`;
    chip.querySelector("button").onclick = () => { pending.splice(i, 1); paintTray(); };
    tray.append(chip);
  });
}

/* --------------------------------------------------------------- paint -- */

function paint(msg) {
  const row = document.createElement("article");
  row.className = `msg ${msg.role}`;
  const who = msg.role === "user" ? "you" : msg.role === "assistant" ? "k3" : msg.role;
  row.innerHTML = `
    <div class="rail"><span class="who">${who}</span><span class="meta">${msg.meta || stamp()}</span></div>
    <div class="body"></div>`;
  const body = row.querySelector(".body");

  if (msg.reasoning) body.append(traceNode(msg.reasoning));
  if (msg.images?.length) body.append(shotsNode(msg.images, msg.made));
  const prose = document.createElement("div");
  prose.className = "prose";
  prose.innerHTML = render(msg.text || "");
  body.append(prose);

  const tools = document.createElement("div");
  tools.className = "rowtools";
  if (msg.text) {
    const copy = document.createElement("button");
    copy.textContent = "Copy";
    copy.onclick = async () => {
      await navigator.clipboard.writeText(msg.text);
      copy.textContent = "Copied";
      setTimeout(() => (copy.textContent = "Copy"), 1200);
    };
    tools.append(copy);
  }
  if (msg.made && msg.images?.length) {
    const reuse = document.createElement("button");
    reuse.textContent = "Ask K3 about this";
    reuse.onclick = () => {
      pending.push(...msg.images);
      paintTray();
      document.querySelector('.mode-btn[data-mode="chat"]').click();
      say("Image attached to your next message.");
    };
    tools.append(reuse);
  }
  if (tools.childElementCount) body.append(tools);
  thread.append(row);
  return row;
}

function traceNode(text) {
  const box = document.createElement("details");
  box.className = "think";
  box.open = $("#show-thinking").checked;
  box.innerHTML = `<summary>Reasoning</summary><div class="trace"></div>`;
  box.querySelector(".trace").textContent = text;
  return box;
}

function shotsNode(urls, made) {
  const wrap = document.createElement("div");
  wrap.className = `shots${made ? " made" : ""}`;
  urls.forEach((url, i) => {
    const img = new Image();
    img.src = url;
    img.alt = made ? `Generated image ${i + 1}` : `Attached image ${i + 1}`;
    img.loading = "lazy";
    wrap.append(img);
  });
  return wrap;
}

/* ------------------------------------------------------------- payload -- */

function payload() {
  const out = [];
  const system = $("#sys-prompt").value.trim();
  if (system) out.push({ role: "system", content: system });
  for (const m of messages) {
    if (m.role === "assistant" && m.made) {
      out.push({ role: "assistant", content: m.text || "[image generated]" });
      continue;
    }
    if (m.images?.length) {
      const parts = m.images.map((url) => ({ type: "image_url", image_url: { url } }));
      if (m.text) parts.push({ type: "text", text: m.text });
      out.push({ role: m.role, content: parts });
    } else {
      out.push({ role: m.role, content: m.text });
    }
  }
  return out;
}

/* ---------------------------------------------------------------- chat -- */

async function ask() {
  const text = input.value.trim();
  if (!text && pending.length === 0) return;

  $("#empty")?.remove();
  const user = { role: "user", text, images: pending, meta: stamp() };
  messages.push(user);
  paint(user);
  pending = [];
  paintTray();
  input.value = "";
  autosize();
  scrollDown(true);

  const reply = { role: "assistant", text: "", reasoning: "", images: [], meta: "streaming…" };
  const row = paint(reply);
  const body = row.querySelector(".body");
  const prose = row.querySelector(".prose");
  const meta = row.querySelector(".rail .meta");
  const cursor = document.createElement("span");
  cursor.className = "cursor";
  prose.append(cursor);
  let trace = null;

  busy(true);
  controller = new AbortController();
  const maxTokens = parseInt($("#max-tokens").value, 10);

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: controller.signal,
      body: JSON.stringify({
        messages: payload(),
        temperature: parseFloat($("#temp").value),
        ...(maxTokens ? { max_tokens: maxTokens } : {}),
      }),
    });
    if (!res.ok || !res.body) throw new Error(`Gateway returned ${res.status}.`);

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const frames = buffer.split("\n\n");
      buffer = frames.pop();
      for (const frame of frames) {
        const line = frame.split("\n").find((l) => l.startsWith("data:"));
        if (!line) continue;
        const evt = JSON.parse(line.slice(5).trim());

        if (evt.type === "reasoning") {
          reply.reasoning += evt.text;
          if (!trace) { trace = traceNode(""); body.prepend(trace); }
          trace.querySelector(".trace").textContent = reply.reasoning;
        } else if (evt.type === "delta") {
          reply.text += evt.text;
          prose.innerHTML = render(reply.text);
          prose.append(cursor);
          scrollDown();
        } else if (evt.type === "notice") {
          say(evt.message);
        } else if (evt.type === "error") {
          throw new Error(evt.message);
        } else if (evt.type === "done") {
          const tok = evt.usage?.completion_tokens;
          const rate = tok && evt.elapsed_ms ? (tok / (evt.elapsed_ms / 1000)).toFixed(1) : null;
          reply.meta = [stamp(),
            tok ? `${tok} tok` : null,
            rate ? `${rate} tok/s` : null,
            evt.ttft_ms ? `ttft ${evt.ttft_ms}ms` : null,
          ].filter(Boolean).join("\n");
        }
      }
    }
    cursor.remove();
    meta.textContent = reply.meta === "streaming…" ? stamp() : reply.meta;
    messages.push(reply);
    save();
    row.replaceWith(paint(reply));
  } catch (err) {
    cursor.remove();
    meta.textContent = stamp();
    if (err.name === "AbortError") {
      if (reply.text) { messages.push(reply); save(); }
      say("Stopped.");
    } else {
      const fail = document.createElement("div");
      fail.className = "fail";
      fail.textContent = err.message;
      body.append(fail);
      say(err.message, true);
    }
  } finally {
    busy(false);
    scrollDown();
  }
}

/* ------------------------------------------------------- image drawing -- */

async function draw() {
  const prompt = input.value.trim();
  if (!prompt) return;
  $("#empty")?.remove();

  const user = { role: "user", text: prompt, images: [], meta: stamp() };
  messages.push(user);
  paint(user);
  input.value = "";
  autosize();

  const reply = { role: "assistant", text: "", images: [], made: true, meta: "drawing…" };
  const row = paint(reply);
  const body = row.querySelector(".body");
  busy(true);
  controller = new AbortController();

  try {
    const res = await fetch("/api/images", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: controller.signal,
      body: JSON.stringify({ prompt, size: $("#img-size").value, n: 1 }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || `Image server returned ${res.status}.`);

    reply.images = data.images.map((i) => i.url);
    reply.text = `Drawn with ${data.model}.`;
    reply.meta = [stamp(), `${(data.elapsed_ms / 1000).toFixed(1)}s`].join("\n");
    messages.push(reply);
    save();
    row.replaceWith(paint(reply));
    thread.lastElementChild.scrollIntoView({ block: "nearest" });
  } catch (err) {
    const fail = document.createElement("div");
    fail.className = "fail";
    fail.textContent = err.name === "AbortError" ? "Stopped." : err.message;
    body.append(fail);
    row.querySelector(".rail .meta").textContent = stamp();
  } finally {
    busy(false);
    scrollDown();
  }
}

function busy(on) {
  sendBtn.hidden = on;
  stopBtn.hidden = !on;
  input.disabled = on;
  if (!on) { controller = null; input.focus(); }
}

/* --------------------------------------------------------------- wiring -- */

$("#composer").addEventListener("submit", (e) => {
  e.preventDefault();
  (mode === "image" ? draw : ask)();
});

stopBtn.onclick = () => controller?.abort();

input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    $("#composer").requestSubmit();
  }
});
input.addEventListener("input", autosize);

$("#file").addEventListener("change", (e) => { stage(e.target.files); e.target.value = ""; });

document.addEventListener("paste", (e) => {
  const files = [...(e.clipboardData?.files || [])];
  if (files.length) { e.preventDefault(); stage(files); }
});

["dragover", "dragleave", "drop"].forEach((type) => {
  inputRow.addEventListener(type, (e) => {
    e.preventDefault();
    inputRow.classList.toggle("dragging", type === "dragover");
    if (type === "drop") stage(e.dataTransfer.files);
  });
});

document.querySelectorAll(".mode-btn").forEach((btn) => {
  btn.onclick = () => {
    mode = btn.dataset.mode;
    document.querySelectorAll(".mode-btn").forEach((b) => b.classList.toggle("is-on", b === btn));
    document.body.classList.toggle("mode-image", mode === "image");
    input.placeholder = mode === "image"
      ? "Describe the image to draw."
      : "Ask anything. Enter sends, Shift+Enter adds a line.";
    say(mode === "image" ? `Drawing with ${cfg.image_model}.` : "");
    input.focus();
  };
});

$("#btn-settings").onclick = (e) => {
  const panel = $("#settings");
  panel.hidden = !panel.hidden;
  e.currentTarget.setAttribute("aria-expanded", String(!panel.hidden));
};

$("#btn-new").onclick = () => {
  if (messages.length && !confirm("Clear this conversation?")) return;
  messages = [];
  pending = [];
  paintTray();
  localStorage.removeItem(STORE_KEY);
  thread.innerHTML = "";
  location.reload();
};

$("#temp").addEventListener("input", (e) => ($("#temp-val").textContent = e.target.value));

$("#show-thinking").addEventListener("change", (e) => {
  document.querySelectorAll("details.think").forEach((d) => (d.open = e.target.checked));
});

thread.addEventListener("click", (e) => {
  if (!e.target.classList.contains("starter")) return;
  input.value = e.target.textContent;
  if (/^draw|illustrat/i.test(input.value) && cfg.image_enabled) $("#mode-image").click();
  autosize();
  input.focus();
});

/* --------------------------------------------------------------- start -- */

(async function start() {
  try {
    cfg = await (await fetch("/api/config")).json();
    $("#model-tag").textContent = cfg.chat_model;
    $("#temp").value = cfg.default_temperature;
    $("#temp-val").textContent = cfg.default_temperature;
    if (!cfg.image_enabled) {
      const btn = $("#mode-image");
      btn.disabled = true;
      btn.title = "Set IMAGE_BASE_URL to enable image generation.";
    }
  } catch {
    say("Cannot load app config.", true);
  }
  try {
    const ready = await fetch("/readyz");
    $("#status-dot").className = `dot ${ready.ok ? "ok" : "down"}`;
    $("#status-dot").title = ready.ok ? "SGLang reachable" : "SGLang unreachable";
    if (!ready.ok) say("SGLang is not answering. Check SGLANG_BASE_URL and the model server.", true);
  } catch {
    $("#status-dot").className = "dot down";
  }
  restore();
  autosize();
  input.focus();
})();
