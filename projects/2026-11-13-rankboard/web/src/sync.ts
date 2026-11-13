// Feed follower: pulls /changes pages into a BoardState, reloading the
// snapshot when the server says the cursor is too old (410 after compaction).
import { Api, ApiError } from "./api.js";
import type { BoardState } from "./board.js";

export async function pull(api: Api, state: BoardState): Promise<"ok" | "reloaded"> {
  for (;;) {
    try {
      const page = await api.changes(state.boardId, state.cursor);
      state.applyChanges(page.changes, page.cursor);
      if (!page.has_more) return "ok";
    } catch (e) {
      if (e instanceof ApiError && e.status === 410) {
        state.loadSnapshot(await api.snapshot(state.boardId));
        return "reloaded";
      }
      throw e;
    }
  }
}
