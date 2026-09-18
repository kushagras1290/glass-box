/**
 * The field — the single full-bleed subject of the whole site (cinematic-web tier 2).
 *
 * The subject is a probability distribution, which has no photoreal form, so procedural
 * canvas rendering is the correct choice here rather than the janky one (skill Part 4,
 * rule zero). Everything eases in elapsed milliseconds so the feel is identical at
 * 60/90/120Hz (§3.11), DPR is capped (§3.12), and the loop pauses when hidden (Part 5).
 */
import type { RGB } from '../story/chapters';
import type { FieldTarget } from '../story/director';

const GOLDEN_ANGLE = Math.PI * (3 - Math.sqrt(5));
const TAU_CELL_MS = 150;
const TAU_FRAME_MS = 320;
const R_MIN = 0.06;
const R_MAX = 0.42;
const DUST_COUNT = 70;

interface Cell {
  readonly tokenId: number;
  text: string;
  p: number;
  rf: number;
  angle: number;
  size: number;
  alpha: number;
  targetRf: number;
  targetAngle: number;
  targetSize: number;
  targetAlpha: number;
  chosen: boolean;
  ignite: number;
}

interface Dust {
  x: number;
  y: number;
  z: number;
  vx: number;
  vy: number;
}

interface Frame {
  focusX: number;
  focusY: number;
  spread: number;
  dim: number;
  point: number;
  r: number;
  g: number;
  b: number;
}

export interface FieldOptions {
  readonly dprCap: number;
  readonly reducedMotion: boolean;
}

const ease = (current: number, target: number, blend: number): number => current + (target - current) * blend;

function shortestAngle(from: number, to: number): number {
  return Math.atan2(Math.sin(to - from), Math.cos(to - from));
}

export class FieldScene {
  private readonly ctx: CanvasRenderingContext2D | null;
  private readonly cells = new Map<number, Cell>();
  private readonly dust: Dust[];
  private target: FieldTarget | null = null;
  private frame: Frame = { focusX: 0.5, focusY: 0.5, spread: 1, dim: 1, point: 0.6, r: 168, g: 196, b: 204 };
  private raf: number | null = null;
  private last = 0;
  private width = 0;
  private height = 0;
  private visible = !document.hidden;
  private readonly canvas: HTMLCanvasElement;
  private readonly options: FieldOptions;

  constructor(canvas: HTMLCanvasElement, options: FieldOptions) {
    this.canvas = canvas;
    this.options = options;
    this.ctx = canvas.getContext('2d');
    this.dust = Array.from({ length: DUST_COUNT }, (_, i) => {
      // Deterministic scatter so the atmosphere is identical on every load.
      const f = (n: number) => { const v = Math.sin((i + 1) * n) * 43758.5453; return v - Math.floor(v); };
      return {
        x: f(12.9898),
        y: f(78.233),
        z: 0.2 + f(39.425) * 0.8,
        vx: (f(5.398) - 0.5) * 0.000012,
        vy: (f(9.137) - 0.5) * 0.00001,
      };
    });
    document.addEventListener('visibilitychange', this.onVisibility);
    this.resize();
  }

  private onVisibility = (): void => {
    this.visible = !document.hidden;
    if (this.visible) this.start();
    else this.stop();
  };

  resize(): void {
    const rect = this.canvas.getBoundingClientRect();
    this.width = rect.width;
    this.height = rect.height;
    const dpr = Math.min(window.devicePixelRatio || 1, this.options.dprCap);
    this.canvas.width = Math.max(1, Math.round(this.width * dpr));
    this.canvas.height = Math.max(1, Math.round(this.height * dpr));
    this.ctx?.setTransform(dpr, 0, 0, dpr, 0, 0);
    if (this.options.reducedMotion) this.draw(0);
  }

  setTarget(target: FieldTarget): void {
    this.target = target;
    const seen = new Set<number>();

    target.cells.forEach((cell, rank) => {
      seen.add(cell.tokenId);
      // p^0.45 compresses the range so a 0.97 token doesn't sit alone with every other
      // candidate smeared against the rim.
      const targetRf = R_MIN + (1 - Math.pow(cell.p, 0.45)) * (R_MAX - R_MIN);
      const targetAngle = GOLDEN_ANGLE * rank - Math.PI / 2;
      const targetSize = 4 + Math.sqrt(cell.p) * 58;
      const targetAlpha = cell.p < 0.0005 ? 0 : 0.25 + Math.min(1, cell.p * 1.6) * 0.75;
      const chosen = cell.tokenId === target.chosenId;

      const existing = this.cells.get(cell.tokenId);
      if (existing) {
        existing.text = cell.text;
        existing.p = cell.p;
        existing.targetRf = targetRf;
        existing.targetAngle = targetAngle;
        existing.targetSize = targetSize;
        existing.targetAlpha = targetAlpha;
        if (chosen && !existing.chosen) existing.ignite = 1;
        existing.chosen = chosen;
      } else {
        this.cells.set(cell.tokenId, {
          tokenId: cell.tokenId,
          text: cell.text,
          p: cell.p,
          rf: this.options.reducedMotion ? targetRf : 0,
          angle: targetAngle,
          size: this.options.reducedMotion ? targetSize : 0,
          alpha: this.options.reducedMotion ? targetAlpha : 0,
          targetRf,
          targetAngle,
          targetSize,
          targetAlpha,
          chosen,
          ignite: chosen ? 1 : 0,
        });
      }
    });

    for (const cell of this.cells.values()) {
      if (!seen.has(cell.tokenId)) {
        cell.targetAlpha = 0;
        cell.targetSize = 0;
      }
    }

    if (this.options.reducedMotion) this.draw(0);
    else this.start();
  }

  start(): void {
    if (this.raf !== null || !this.visible || this.options.reducedMotion) return;
    this.last = performance.now();
    this.raf = requestAnimationFrame(this.tick);
  }

  stop(): void {
    if (this.raf !== null) cancelAnimationFrame(this.raf);
    this.raf = null;
  }

  private tick = (time: number): void => {
    const elapsed = Math.min(64, time - this.last);
    this.last = time;
    this.draw(elapsed, time);
    this.raf = requestAnimationFrame(this.tick);
  };

  /** elapsed === 0 means "snap to target" — the reduced-motion path. */
  private draw(elapsed: number, time = 0): void {
    const ctx = this.ctx;
    const target = this.target;
    if (!ctx) return;

    const snap = elapsed === 0;
    const cellBlend = snap ? 1 : 1 - Math.exp(-elapsed / TAU_CELL_MS);
    const frameBlend = snap ? 1 : 1 - Math.exp(-elapsed / TAU_FRAME_MS);
    const f = this.frame;

    if (target) {
      f.focusX = ease(f.focusX, target.focusX, frameBlend);
      f.focusY = ease(f.focusY, target.focusY, frameBlend);
      f.spread = ease(f.spread, target.spread, frameBlend);
      f.dim = ease(f.dim, target.dim, frameBlend);
      f.point = ease(f.point, target.point, frameBlend);
      f.r = ease(f.r, target.accent[0], frameBlend);
      f.g = ease(f.g, target.accent[1], frameBlend);
      f.b = ease(f.b, target.accent[2], frameBlend);
    }

    const accent: RGB = [Math.round(f.r), Math.round(f.g), Math.round(f.b)];
    const rgb = `${accent[0]}, ${accent[1]}, ${accent[2]}`;
    const minDim = Math.min(this.width, this.height);
    const scale = Math.min(1.25, Math.max(0.55, minDim / 760));
    const cx = this.width * f.focusX;
    const cy = this.height * f.focusY;

    ctx.clearRect(0, 0, this.width, this.height);

    // Atmosphere — subdued particulate dust, tinted by the chapter accent.
    for (const d of this.dust) {
      if (!snap) {
        d.x += d.vx * elapsed;
        d.y += d.vy * elapsed;
        if (d.x < -0.02) d.x = 1.02;
        if (d.x > 1.02) d.x = -0.02;
        if (d.y < -0.02) d.y = 1.02;
        if (d.y > 1.02) d.y = -0.02;
      }
      ctx.beginPath();
      ctx.arc(d.x * this.width, d.y * this.height, 0.5 + d.z * 1.1, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${rgb}, ${(0.05 + d.z * 0.16).toFixed(3)})`;
      ctx.fill();
    }

    // The single point — dormant, loading, home.
    if (f.point > 0.01) {
      const pulse = snap ? 0 : Math.sin(time * 0.0016) * 0.12;
      const radius = (3 + f.point * 9) * scale * (1 + pulse);
      const halo = ctx.createRadialGradient(cx, cy, 0, cx, cy, radius * 7);
      halo.addColorStop(0, `rgba(${rgb}, ${(f.point * 0.42).toFixed(3)})`);
      halo.addColorStop(1, `rgba(${rgb}, 0)`);
      ctx.fillStyle = halo;
      ctx.beginPath();
      ctx.arc(cx, cy, radius * 7, 0, Math.PI * 2);
      ctx.fill();
      ctx.beginPath();
      ctx.arc(cx, cy, radius, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(245, 250, 251, ${Math.min(1, f.point * 1.2).toFixed(3)})`;
      ctx.fill();
    }

    // The distribution.
    for (const [id, cell] of this.cells) {
      cell.rf = ease(cell.rf, cell.targetRf, cellBlend);
      cell.angle += shortestAngle(cell.angle, cell.targetAngle) * cellBlend;
      cell.size = ease(cell.size, cell.targetSize, cellBlend);
      cell.alpha = ease(cell.alpha, cell.targetAlpha, cellBlend);
      if (cell.ignite > 0) cell.ignite = snap ? 0 : Math.max(0, cell.ignite - elapsed / 520);

      if (cell.targetAlpha === 0 && cell.alpha < 0.01) {
        this.cells.delete(id);
        continue;
      }

      const alpha = cell.alpha * f.dim;
      if (alpha < 0.01) continue;
      const r = cell.rf * f.spread * minDim;
      const x = cx + Math.cos(cell.angle) * r;
      const y = cy + Math.sin(cell.angle) * r;
      const size = Math.max(0.5, cell.size * scale);

      ctx.beginPath();
      ctx.arc(x, y, size, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${rgb}, ${(alpha * 0.16).toFixed(3)})`;
      ctx.fill();
      ctx.lineWidth = cell.chosen ? 1.5 : 0.8;
      ctx.strokeStyle = cell.chosen
        ? `rgba(245, 214, 180, ${Math.min(1, alpha * (0.7 + cell.ignite * 0.3)).toFixed(3)})`
        : `rgba(${rgb}, ${(alpha * 0.6).toFixed(3)})`;
      if (cell.ignite > 0 && !snap) {
        ctx.shadowBlur = 22 * cell.ignite;
        ctx.shadowColor = `rgba(245, 214, 180, ${cell.ignite.toFixed(3)})`;
      }
      ctx.stroke();
      ctx.shadowBlur = 0;

      if (size > 8 && alpha > 0.3) {
        const fontSize = Math.round(Math.min(15, 8 + size * 0.2));
        ctx.font = `${fontSize}px ui-monospace, SFMono-Regular, Menlo, monospace`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillStyle = `rgba(236, 243, 245, ${Math.min(1, alpha).toFixed(3)})`;
        const label = cell.text.length > 10 ? `${cell.text.slice(0, 9)}…` : cell.text;
        const showPct = Boolean(target?.showPercent) && cell.p >= 0.04 && size > 14;
        ctx.fillText(label, x, showPct ? y - fontSize * 0.45 : y);
        if (showPct) {
          ctx.font = `${Math.max(9, fontSize - 3)}px ui-monospace, SFMono-Regular, Menlo, monospace`;
          ctx.fillStyle = `rgba(${rgb}, ${Math.min(1, alpha).toFixed(3)})`;
          ctx.fillText(`${(cell.p * 100).toFixed(cell.p >= 0.1 ? 0 : 1)}%`, x, y + fontSize * 0.75);
        }
      }
    }
  }

  destroy(): void {
    this.stop();
    document.removeEventListener('visibilitychange', this.onVisibility);
    this.cells.clear();
  }
}
