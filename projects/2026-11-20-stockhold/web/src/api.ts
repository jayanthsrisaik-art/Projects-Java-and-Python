// HTTP client with safe retries.
//
// A purchase *intent* (one click on "Reserve") gets one Idempotency-Key, and
// every retry of that intent reuses it. If the server committed the hold but the
// response was lost, the retry replays the original hold instead of creating a
// second one. Retries happen only for network failures and 5xx/429; a business
// answer (409 sold out, 402 declined, 410 expired) is final.

import { clockOffsetMs, type Hold, type ProductView } from "./hold.js";

export type FetchLike = (url: string, init?: RequestInit) => Promise<Response>;

export interface ApiResult<T> {
  ok: boolean;
  status: number;
  body: T & { error?: string; message?: string };
  replayed: boolean;
  attempts: number;
}

export interface ClientOptions {
  base?: string;
  fetch?: FetchLike;
  sleep?: (ms: number) => Promise<void>;
  now?: () => number;
  maxAttempts?: number;
  backoffMs?: number;
}

export function newIdempotencyKey(): string {
  return `intent-${globalThis.crypto.randomUUID()}`;
}

export class ApiClient {
  private readonly base: string;
  private readonly fetchImpl: FetchLike;
  private readonly sleep: (ms: number) => Promise<void>;
  private readonly now: () => number;
  private readonly maxAttempts: number;
  private readonly backoffMs: number;
  /** Latest estimate of (server clock - browser clock) in ms. */
  offsetMs = 0;

  constructor(public customerId: string, opts: ClientOptions = {}) {
    this.base = opts.base ?? "";
    this.fetchImpl = opts.fetch ?? ((u, i) => fetch(u, i));
    this.sleep = opts.sleep ?? ((ms) => new Promise((r) => setTimeout(r, ms)));
    this.now = opts.now ?? (() => Date.now());
    this.maxAttempts = opts.maxAttempts ?? 4;
    this.backoffMs = opts.backoffMs ?? 250;
  }

  private async request<T>(
    method: string,
    path: string,
    body?: unknown,
    extra: Record<string, string> = {},
  ): Promise<ApiResult<T>> {
    const headers: Record<string, string> = { "X-Customer-Id": this.customerId, ...extra };
    if (body !== undefined) headers["Content-Type"] = "application/json";
    let lastStatus = 0;
    for (let attempt = 1; attempt <= this.maxAttempts; attempt++) {
      const sent = this.now();
      let res: Response;
      try {
        res = await this.fetchImpl(this.base + path, {
          method,
          headers,
          body: body === undefined ? undefined : JSON.stringify(body),
        });
      } catch {
        lastStatus = 0;
        if (attempt < this.maxAttempts) await this.sleep(this.backoffMs * 2 ** (attempt - 1));
        continue;
      }
      if ((res.status >= 500 || res.status === 429) && attempt < this.maxAttempts) {
        lastStatus = res.status;
        await this.sleep(this.backoffMs * 2 ** (attempt - 1));
        continue;
      }
      const json = (await res.json()) as T & { error?: string; server_time?: number };
      if (typeof json.server_time === "number") {
        this.offsetMs = clockOffsetMs(json.server_time, sent, this.now());
      }
      return {
        ok: res.ok,
        status: res.status,
        body: json,
        replayed: res.headers.get("Idempotent-Replayed") === "true",
        attempts: attempt,
      };
    }
    return {
      ok: false,
      status: lastStatus,
      body: { error: "network" } as T & { error: string },
      replayed: false,
      attempts: this.maxAttempts,
    };
  }

  products(): Promise<ApiResult<{ products: ProductView[]; server_time: number }>> {
    return this.request("GET", "/api/products");
  }

  /** The caller creates the key once per user intent and passes it in. */
  reserve(sku: string, qty: number, idempotencyKey: string): Promise<ApiResult<Hold>> {
    return this.request("POST", "/api/holds", { sku, qty }, { "Idempotency-Key": idempotencyKey });
  }

  hold(id: string): Promise<ApiResult<Hold>> {
    return this.request("GET", `/api/holds/${encodeURIComponent(id)}`);
  }

  confirm(id: string, paymentToken: string): Promise<ApiResult<{ id: string; amount_cents: number; hold_id: string }>> {
    // Safe to retry without a key: confirming a confirmed hold returns the same order.
    return this.request("POST", `/api/holds/${encodeURIComponent(id)}/confirm`, { payment_token: paymentToken });
  }

  release(id: string): Promise<ApiResult<Hold>> {
    return this.request("POST", `/api/holds/${encodeURIComponent(id)}/release`);
  }
}
