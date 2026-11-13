// Browser UI: board picker, columns, drag-and-drop with optimistic placement,
// inline title edits with If-Match, and a 1.5 s change-feed poll.
import { Api, ApiError, newCardId } from "./api.js";
import { BoardState, type Card } from "./board.js";
import { pull } from "./sync.js";

const api = new Api();
const state = new BoardState();
const $ = <T extends HTMLElement>(sel: string) => document.querySelector(sel) as T;
let dragging: string | null = null;

function toast(msg: string): void {
  const el = $("#toast");
  el.textContent = msg;
  el.hidden = false;
  window.setTimeout(() => (el.hidden = true), 3500);
}

function cardEl(card: Card): HTMLElement {
  const li = document.createElement("li");
  li.className = "card";
  li.draggable = true;
  li.dataset.id = card.id;
  const title = document.createElement("span");
  title.className = "title";
  title.textContent = card.title;
  title.title = "double-click to rename";
  const rank = document.createElement("code");
  rank.className = "rank";
  rank.textContent = card.rank;
  const del = document.createElement("button");
  del.className = "del";
  del.textContent = "×";
  del.title = "delete";
  li.append(title, rank, del);
  li.addEventListener("dragstart", (e) => {
    dragging = card.id;
    li.classList.add("dragging");
    e.dataTransfer?.setData("text/plain", card.id);
  });
  li.addEventListener("dragend", () => {
    dragging = null;
    li.classList.remove("dragging");
  });
  title.addEventListener("dblclick", () => rename(card.id));
  del.addEventListener("click", () => remove(card.id));
  return li;
}

function render(): void {
  $("#board-name").textContent = state.name;
  $("#status").textContent = `cursor ${state.cursor}` + (state.pendingCount() ? ` · ${state.pendingCount()} pending` : "");
  const root = $("#columns");
  root.replaceChildren();
  for (const col of state.columns) {
    const section = document.createElement("section");
    section.className = "column";
    const h = document.createElement("h2");
    h.textContent = col.name;
    const list = document.createElement("ul");
    list.dataset.column = col.id;
    for (const c of state.cardsIn(col.id)) list.append(cardEl(c));
    list.addEventListener("dragover", (e) => {
      e.preventDefault();
      list.classList.add("over");
    });
    list.addEventListener("dragleave", () => list.classList.remove("over"));
    list.addEventListener("drop", (e) => {
      e.preventDefault();
      list.classList.remove("over");
      if (dragging) drop(dragging, col.id, list, e.clientY);
    });
    const form = document.createElement("form");
    form.innerHTML = `<input name="title" placeholder="Add a card" autocomplete="off" maxlength="500"><button>Add</button>`;
    form.addEventListener("submit", (e) => {
      e.preventDefault();
      const input = form.elements.namedItem("title") as HTMLInputElement;
      const title = input.value.trim();
      if (title) void create(col.id, title);
      input.value = "";
    });
    section.append(h, list, form);
    root.append(section);
  }
}

/** Work out the neighbours under the pointer, as this client sees the column. */
function neighbours(list: HTMLElement, y: number, moving: string): [string | null, string | null] {
  const items = [...list.querySelectorAll<HTMLElement>("li.card")].filter((li) => li.dataset.id !== moving);
  let after: string | null = null;
  for (const li of items) {
    const box = li.getBoundingClientRect();
    if (y < box.top + box.height / 2) return [after, li.dataset.id!];
    after = li.dataset.id!;
  }
  return [after, null];
}

async function drop(id: string, columnId: string, list: HTMLElement, y: number): Promise<void> {
  const [after, before] = neighbours(list, y, id);
  const token = state.moveOptimistic(id, columnId, after, before);
  render();
  try {
    const res = await api.moveCard(id, columnId, after, before);
    state.confirm(id, token, res.card);
    if (res.placement === "fallback_end") toast("The cards you dropped next to moved; placed at the end.");
  } catch (e) {
    state.rollback(id, token);
    toast(`Move failed: ${(e as Error).message}`);
  }
  render();
}

async function create(columnId: string, title: string): Promise<void> {
  const id = newCardId();
  const token = state.createOptimistic(id, columnId, title);
  render();
  for (let attempt = 0; ; attempt++) {
    try {
      const res = await api.createCard(state.boardId, { id, column_id: columnId, title, after_id: null, before_id: null });
      state.confirm(id, token, res.card);
      break;
    } catch (e) {
      // network errors are retried with the same id: the server deduplicates
      if (!(e instanceof ApiError) && attempt < 2) continue;
      state.rollback(id, token);
      toast(`Could not add card: ${(e as Error).message}`);
      break;
    }
  }
  render();
}

async function rename(id: string): Promise<void> {
  const card = state.get(id);
  if (!card) return;
  const title = window.prompt("Rename card", card.title)?.trim();
  if (!title || title === card.title) return;
  const token = state.editOptimistic(id, title);
  render();
  try {
    state.confirm(id, token, await api.editCard(id, card.version, title));
  } catch (e) {
    if (e instanceof ApiError && e.status === 412) {
      state.rollback(id, token, e.body.current);
      toast(`Someone else changed this card first: now "${e.body.current.title}". Your edit was not saved.`);
    } else {
      state.rollback(id, token);
      toast(`Rename failed: ${(e as Error).message}`);
    }
  }
  render();
}

async function remove(id: string): Promise<void> {
  const card = state.get(id);
  if (!card) return;
  const token = state.deleteOptimistic(id);
  render();
  try {
    await api.deleteCard(id, card.version);
    state.confirm(id, token, null);
  } catch (e) {
    state.rollback(id, token, e instanceof ApiError && e.status === 412 ? e.body.current : undefined);
    toast(e instanceof ApiError && e.status === 412 ? "Card changed since you loaded it; not deleted." : `Delete failed: ${(e as Error).message}`);
  }
  render();
}

async function openBoard(id: string): Promise<void> {
  state.loadSnapshot(await api.snapshot(id));
  history.replaceState(null, "", `#${id}`);
  render();
}

async function main(): Promise<void> {
  let id = location.hash.slice(1);
  if (!id) {
    const boards = await api.boards();
    id = boards.length ? boards[0].id : (await api.createBoard("My board")).board.id;
  }
  await openBoard(id);
  window.setInterval(async () => {
    if (dragging) return; // don't re-render under the user's pointer
    try {
      if ((await pull(api, state)) === "reloaded") toast("Reloaded board (history was compacted).");
      render();
    } catch {
      $("#status").textContent = "offline – retrying";
    }
  }, 1500);
}

void main();
