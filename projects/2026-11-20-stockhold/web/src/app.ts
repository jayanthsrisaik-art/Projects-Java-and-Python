// Browser UI: product list with live stock, reserve -> countdown -> pay / release.
// All server data is written with textContent (never innerHTML).

import { ApiClient, newIdempotencyKey } from "./api.js";
import {
  explain,
  formatCountdown,
  formatPrice,
  remainingMs,
  stockBar,
  stockLabel,
  type Hold,
  type ProductView,
} from "./hold.js";

const STORE_KEY = "stockhold.session";

interface Saved {
  customer: string;
  holds: string[];
}

function load(): Saved {
  try {
    const raw = localStorage.getItem(STORE_KEY);
    if (raw) return JSON.parse(raw) as Saved;
  } catch {
    /* storage unavailable: fall through to a fresh session */
  }
  return { customer: `shopper-${Math.random().toString(36).slice(2, 8)}`, holds: [] };
}

function save(s: Saved): void {
  try {
    localStorage.setItem(STORE_KEY, JSON.stringify(s));
  } catch {
    /* non-essential */
  }
}

function el<K extends keyof HTMLElementTagNameMap>(tag: K, cls?: string, text?: string): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

const state = load();
const api = new ApiClient(state.customer);
const holds = new Map<string, Hold>();
let products: ProductView[] = [];
const pending = new Set<string>();
// Form choices survive the periodic re-render.
const chosenQty = new Map<string, string>();
const chosenCard = new Map<string, string>();

const root = document.getElementById("app")!;
const flash = el("p", "flash");

function say(text: string, kind: "ok" | "err" = "ok"): void {
  flash.textContent = text;
  flash.dataset.kind = kind;
}

function persistHolds(): void {
  state.holds = [...holds.values()].filter((h) => h.status === "active").map((h) => h.id);
  save(state);
}

async function refreshProducts(): Promise<void> {
  const r = await api.products();
  if (r.ok) products = r.body.products;
  render();
}

async function reserve(sku: string, qty: number): Promise<void> {
  if (pending.has(sku)) return; // ignore double-clicks while a request is in flight
  pending.add(sku);
  render();
  const key = newIdempotencyKey(); // one key per intent, reused by every retry
  try {
    const r = await api.reserve(sku, qty, key);
    if (r.ok) {
      holds.set(r.body.id, r.body);
      persistHolds();
      say(r.replayed ? "Reserved (recovered after a retry)." : "Reserved! Pay before the timer runs out.");
    } else {
      say(explain(r.body.error), "err");
    }
  } finally {
    pending.delete(sku);
    await refreshProducts();
  }
}

async function pay(hold: Hold, token: string): Promise<void> {
  if (pending.has(hold.id)) return;
  pending.add(hold.id);
  render();
  try {
    const r = await api.confirm(hold.id, token);
    if (r.ok) {
      holds.set(hold.id, { ...hold, status: "confirmed", order_id: r.body.id });
      say(`Paid ${formatPrice(r.body.amount_cents)}. Order ${r.body.id}.`);
    } else {
      if (r.body.error === "hold_expired") holds.set(hold.id, { ...hold, status: "expired" });
      say(explain(r.body.error), "err");
    }
    persistHolds();
  } finally {
    pending.delete(hold.id);
    await refreshProducts();
  }
}

async function release(hold: Hold): Promise<void> {
  const r = await api.release(hold.id);
  if (r.ok) holds.set(hold.id, r.body);
  say(r.ok ? "Released. The item is back on sale." : explain(r.body.error), r.ok ? "ok" : "err");
  persistHolds();
  await refreshProducts();
}

function productCard(p: ProductView): HTMLElement {
  const card = el("article", "card");
  card.append(el("h2", undefined, p.name), el("p", "muted", `${p.sku} · ${formatPrice(p.price_cents)} · limit ${p.per_customer_limit}`));
  const bar = el("div", "bar");
  const w = stockBar(p);
  for (const part of ["sold", "held", "available"] as const) {
    const seg = el("span", part);
    seg.style.width = `${w[part]}%`;
    seg.title = `${part}: ${p[part]}`;
    bar.append(seg);
  }
  card.append(bar, el("p", "stock", stockLabel(p)));
  const row = el("div", "row");
  const qty = el("select");
  qty.setAttribute("aria-label", `Quantity for ${p.name}`);
  for (let i = 1; i <= p.per_customer_limit; i++) qty.append(new Option(String(i), String(i)));
  qty.value = chosenQty.get(p.sku) ?? "1";
  qty.addEventListener("change", () => chosenQty.set(p.sku, qty.value));
  const btn = el("button", undefined, pending.has(p.sku) ? "Reserving…" : "Reserve");
  btn.disabled = pending.has(p.sku) || p.available <= 0;
  btn.addEventListener("click", () => void reserve(p.sku, Number(qty.value)));
  row.append(qty, btn);
  card.append(row);
  return card;
}

function holdRow(h: Hold): HTMLElement {
  const li = el("li", `hold ${h.status}`);
  const name = products.find((p) => p.sku === h.sku)?.name ?? h.sku;
  li.append(el("strong", undefined, `${h.qty} × ${name}`));
  if (h.status === "active") {
    const left = remainingMs(h, api.offsetMs, Date.now());
    const timer = el("span", "timer", formatCountdown(left));
    timer.dataset.hold = h.id;
    const card = el("select");
    card.setAttribute("aria-label", "Test card");
    card.append(new Option("Test card: approves", "tok_ok"), new Option("Test card: declines", "tok_decline"));
    card.value = chosenCard.get(h.id) ?? "tok_ok";
    card.addEventListener("change", () => chosenCard.set(h.id, card.value));
    const payBtn = el("button", undefined, pending.has(h.id) ? "Paying…" : "Pay");
    payBtn.disabled = pending.has(h.id);
    payBtn.addEventListener("click", () => void pay(h, card.value));
    const rel = el("button", "secondary", "Release");
    rel.addEventListener("click", () => void release(h));
    li.append(timer, card, payBtn, rel);
  } else {
    li.append(el("span", "muted", h.status === "confirmed" ? `paid · ${h.order_id}` : h.status));
  }
  return li;
}

function render(): void {
  root.replaceChildren();
  const who = el("label", "who", "Customer id ");
  const input = el("input");
  input.value = state.customer;
  input.addEventListener("change", () => {
    state.customer = input.value.trim() || state.customer;
    api.customerId = state.customer;
    holds.clear();
    persistHolds();
    void refreshProducts();
  });
  who.append(input);
  root.append(who, flash);
  const grid = el("section", "grid");
  for (const p of products) grid.append(productCard(p));
  root.append(grid);
  if (holds.size) {
    root.append(el("h2", undefined, "Your holds"));
    const ul = el("ul", "holds");
    for (const h of [...holds.values()].reverse()) ul.append(holdRow(h));
    root.append(ul);
  }
}

// Countdown ticks update text in place so a re-render doesn't steal focus.
setInterval(() => {
  for (const t of document.querySelectorAll<HTMLElement>(".timer")) {
    const h = holds.get(t.dataset.hold ?? "");
    if (!h) continue;
    const left = remainingMs(h, api.offsetMs, Date.now());
    t.textContent = formatCountdown(left);
    if (left === 0 && h.status === "active") {
      holds.set(h.id, { ...h, status: "expired" });
      persistHolds();
      void api.hold(h.id).then((r) => {
        if (r.ok) holds.set(h.id, r.body);
        void refreshProducts();
      });
    }
  }
}, 250);

setInterval(() => void refreshProducts(), 2000);

void (async () => {
  for (const id of state.holds) {
    const r = await api.hold(id);
    if (r.ok) holds.set(id, r.body);
  }
  await refreshProducts();
})();
