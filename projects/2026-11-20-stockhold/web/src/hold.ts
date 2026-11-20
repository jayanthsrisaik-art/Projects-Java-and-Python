// Pure helpers shared by the UI and its tests (no DOM access here).

export interface ProductView {
  sku: string;
  name: string;
  price_cents: number;
  total: number;
  sold: number;
  held: number;
  available: number;
  per_customer_limit: number;
}

export interface Hold {
  id: string;
  sku: string;
  qty: number;
  status: "active" | "confirmed" | "released" | "expired";
  expires_at: number; // server epoch seconds
  server_time: number; // server epoch seconds when the response was produced
  order_id: string | null;
}

/**
 * Offset (ms) to add to the browser clock to approximate the server clock.
 * NTP-style: assume the server stamped the response halfway through the round trip.
 * The browser clock can be minutes off; the countdown must follow the server,
 * which is the one that decides whether a payment arrived in time.
 */
export function clockOffsetMs(serverTimeS: number, sentAtMs: number, receivedAtMs: number): number {
  const midpoint = (sentAtMs + receivedAtMs) / 2;
  return serverTimeS * 1000 - midpoint;
}

export function remainingMs(hold: Pick<Hold, "expires_at">, offsetMs: number, nowMs: number): number {
  return Math.max(0, hold.expires_at * 1000 - (nowMs + offsetMs));
}

export function formatCountdown(ms: number): string {
  const total = Math.ceil(ms / 1000);
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m}:${s.toString().padStart(2, "0")}`;
}

export function formatPrice(cents: number): string {
  const sign = cents < 0 ? "-" : "";
  const abs = Math.abs(cents);
  return `${sign}$${Math.floor(abs / 100)}.${(abs % 100).toString().padStart(2, "0")}`;
}

/** Percent widths for a stacked sold / held / available bar (sums to 100). */
export function stockBar(p: Pick<ProductView, "total" | "sold" | "held">): { sold: number; held: number; available: number } {
  if (p.total <= 0) return { sold: 0, held: 0, available: 0 };
  const sold = (100 * p.sold) / p.total;
  const held = (100 * p.held) / p.total;
  return { sold, held, available: Math.max(0, 100 - sold - held) };
}

export function stockLabel(p: Pick<ProductView, "total" | "sold" | "held" | "available">): string {
  if (p.available > 0) return `${p.available} of ${p.total} left`;
  if (p.held > 0) return `sold out, ${p.held} in carts. Check back: unpaid holds expire`;
  return "sold out";
}

const MESSAGES: Record<string, string> = {
  sold_out: "Sold out right now. Unpaid holds expire, so stock may come back.",
  limit_exceeded: "You've reached the per-customer limit for this item.",
  hold_expired: "Your hold expired and the item went back on sale.",
  hold_released: "You released this hold.",
  payment_declined: "Payment declined. Your hold is kept: try another card.",
  hold_not_found: "That hold doesn't exist for this customer.",
  idempotency_key_reused: "Request conflict: please try again.",
  network: "Network problem: we retried and couldn't reach the server.",
};

export function explain(error: string | undefined): string {
  return (error && MESSAGES[error]) || "Something went wrong.";
}
