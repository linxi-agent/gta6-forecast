// Forecast GTA VI's launch-day Twitch viewers hour by hour.
//
// Approach: comparable-launch log-ratio quantiles. Every comparable launch in
// the case file contributes its release-day shape as log(viewers_h / viewers_1)
// for each hour h. Shapes are pooled per hour with weights that favour launches
// whose unlock pattern (rolling midnight-local vs one simultaneous global
// unlock) and early-access status match the case, and whose hour-1 scale is
// close to the case's hour-1 print. The forecast is the weighted empirical
// quantiles of those log-ratios, anchored at the case's hour-1 print, with a
// calibrated spread factor. Hours with too little comparable support fall back
// to a flat persistence forecast with a widening lognormal spread.
//
// Deterministic: fixed sort orders, no clock, no randomness. Standard library
// only, no network. Reads every *.in in the input directory (name order) and
// writes one *.out per case.
//
//   MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
const Z = [-1.6448536269514722, -0.6744897501960817, 0, 0.6744897501960817, 1.6448536269514722];
const HOUR_MS = 3600000;

/** Spread multiplier applied to log-quantile deviations from the log-median. */
const SPREAD = 0.9;
/** Log-scale bandwidth of the hour-1 similarity kernel. */
const SCALE_BW = 1.25;
/** Class weight for simultaneous-unlock comparables under a rolling case. */
const SIM_W = 0.25;
/** Class weight for comparables with an unclassified unlock pattern. */
const UNK_W = 0.8;
/** Multiplier when early-access status mismatches between case and comparable. */
const EA_MULT = 0.25;
/** Recency decay in years: older launches weigh less. */
const RECENCY_TAU = 3;
/** Log-shift added to every quantile, ramping in over hours 5-10. */
const SHIFT = 0.15;
/** Minimum effective comparables for a pooled hour; below this, fall back. */
const MIN_POOL = 3;
/** Fallback persistence spread, log units per square-root hour. */
const FALLBACK_SIGMA = 0.15;

interface HourlyPoint {
  from: string;
  viewers: number;
  channels?: number | null;
}

interface Comparable {
  name: string;
  day_start: string;
  unlock?: string;
  platforms?: string;
  hourly: HourlyPoint[];
}

interface CaseFile {
  levels: number[];
  day_start?: string;
  known: Array<{ hour: number; viewers: number }>;
  forecast: Array<{ hour: number }>;
  notes?: string;
  comparables?: Comparable[];
}

interface UnlockFlags {
  rolling: boolean;
  simultaneous: boolean;
  early: boolean;
}

/** Classify an unlock description (or case notes) into pattern flags. */
function classifyUnlock(text: string | undefined): UnlockFlags {
  const t = (text ?? '').toLowerCase();
  return {
    rolling: /midnight local|rolling/.test(t),
    simultaneous: /simultaneous|single global/.test(t),
    early: /early access/.test(t),
  };
}

/** Weight for a comparable given the case's unlock flags. */
function classWeight(caseFlags: UnlockFlags, compFlags: UnlockFlags): number {
  let w: number;
  if (caseFlags.rolling && !caseFlags.simultaneous) {
    if (compFlags.rolling && !compFlags.simultaneous) w = 1.0;
    else if (compFlags.simultaneous) w = SIM_W;
    else w = UNK_W;
  } else if (caseFlags.simultaneous && !caseFlags.rolling) {
    if (compFlags.simultaneous) w = 1.0;
    else if (compFlags.rolling) w = 0.2;
    else w = UNK_W;
  } else {
    w = 1.0;
  }
  if (caseFlags.early !== compFlags.early) w *= EA_MULT;
  return w;
}

/** Release-day hour -> viewers map for one comparable (first point per hour). */
function dayHours(comp: Comparable): Map<number, number> {
  const start = Date.parse(comp.day_start);
  const by = new Map<number, number>();
  if (!Number.isFinite(start)) return by;
  for (const p of comp.hourly ?? []) {
    const t = Date.parse(p.from);
    if (!Number.isFinite(t)) continue;
    const h = Math.round((t - start) / HOUR_MS) + 1;
    if (h >= 1 && h <= 24 && !by.has(h) && typeof p.viewers === 'number') by.set(h, p.viewers);
  }
  return by;
}

/** Weighted quantiles of values at the given levels (midpoint interpolation). */
function weightedQuantiles(pairs: Array<{ v: number; w: number }>, levels: number[]): number[] {
  const sorted = [...pairs].sort((a, b) => a.v - b.v || 0);
  const total = sorted.reduce((s, p) => s + p.w, 0);
  const mids: number[] = [];
  let acc = 0;
  for (const p of sorted) {
    mids.push((acc + p.w / 2) / total);
    acc += p.w;
  }
  return levels.map((tau) => {
    if (tau <= mids[0]) return sorted[0].v;
    const last = sorted.length - 1;
    if (tau >= mids[last]) return sorted[last].v;
    let i = 0;
    while (i < last && mids[i + 1] < tau) i++;
    const f = (tau - mids[i]) / (mids[i + 1] - mids[i]);
    return sorted[i].v + f * (sorted[i + 1].v - sorted[i].v);
  });
}

const round2 = (x: number): number => Math.round(x * 100) / 100;

function clampRow(q: number[]): number[] {
  const out: number[] = [];
  let prev = 0;
  for (const v of q) {
    const c = Math.min(1e9, Math.max(0.01, round2(v)));
    const m = Math.max(prev, c);
    out.push(m);
    prev = m;
  }
  return out;
}

function forecastCase(c: CaseFile): Array<{ hour: number; q: number[] }> {
  const first = c.known.find((k) => k.hour === 1);
  if (first === undefined) throw new Error('case has no hour-1 print');
  const p1 = Math.max(1, first.viewers);
  const logP1 = Math.log(p1);

  // Unlock flags for the case: from notes, else inferred from hour-1 size.
  const flags = classifyUnlock(c.notes);
  const hasPattern = flags.rolling || flags.simultaneous;
  if (!hasPattern) {
    if (p1 >= 10000) flags.rolling = true;
    else if (p1 <= 2000) flags.simultaneous = true;
  }

  // Pool: one entry per comparable with a usable hour 1.
  interface PoolEntry {
    weight: number;
    logRatios: Map<number, number>;
  }
  const caseYear = Date.parse(c.day_start ?? '') / 31557600000;
  const pool: PoolEntry[] = [];
  for (const comp of c.comparables ?? []) {
    const by = dayHours(comp);
    const h1 = by.get(1);
    if (h1 === undefined || !(h1 > 0)) continue;
    const cw = classWeight(flags, classifyUnlock(comp.unlock));
    const sw = Math.max(0.05, Math.exp(-0.5 * Math.pow((logP1 - Math.log(h1)) / SCALE_BW, 2)));
    let rw = 1;
    if (Number.isFinite(RECENCY_TAU) && Number.isFinite(caseYear)) {
      const compYear = Date.parse(comp.day_start) / 31557600000;
      if (Number.isFinite(compYear)) rw = Math.exp(-Math.max(0, caseYear - compYear) / RECENCY_TAU);
    }
    const logRatios = new Map<number, number>();
    for (let h = 2; h <= 24; h++) {
      const v = by.get(h);
      if (v !== undefined && v > 0) logRatios.set(h, Math.log(v / h1));
    }
    pool.push({ weight: cw * sw * rw, logRatios });
  }

  return c.forecast.map(({ hour }) => {
    const pairs: Array<{ v: number; w: number }> = [];
    for (const e of pool) {
      const r = e.logRatios.get(hour);
      if (r !== undefined) pairs.push({ v: r, w: e.weight });
    }
    let q: number[];
    if (pairs.length >= MIN_POOL) {
      const lq = weightedQuantiles(pairs, LEVELS);
      const med = lq[2];
      const shift = SHIFT * Math.min(1, Math.max(0, (hour - 4) / 6));
      q = lq.map((v) => p1 * Math.exp(med + shift + SPREAD * (v - med)));
    } else {
      // Fallback: flat persistence with a widening spread.
      q = Z.map((z) => p1 * Math.exp(z * FALLBACK_SIGMA * Math.sqrt(hour - 1)));
    }
    return { hour, q: clampRow(q) };
  });
}

mkdirSync(OUTPUT_DIR, { recursive: true });
const inputs = readdirSync(INPUT_DIR)
  .filter((f) => f.endsWith('.in'))
  .sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as CaseFile;
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(`${file} asks for levels ${JSON.stringify(c.levels)}`);
  }
  const hours = forecastCase(c);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ hours })}\n`);
}
