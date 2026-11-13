import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

// dist/test/vectors.js -> ../../../samples/key_vectors.json
const here = dirname(fileURLToPath(import.meta.url));
export const vectors = JSON.parse(readFileSync(join(here, "..", "..", "..", "samples", "key_vectors.json"), "utf8")) as {
  between: Array<[string | null, string | null, string]>;
  invalid: Array<[string | null, string | null]>;
  sequence_10: string[];
  placements: Array<{ column: Array<[string, string]>; moving: string | null; after: string | null;
                      before: string | null; rank: string; resolved: string }>;
};
