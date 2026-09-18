/** Config validated at startup — spec §10.4. Crash loudly on a bad value. */
import { ConfigError } from './errors';

export interface AppConfig {
  readonly modelId: string;
  readonly dtype: 'q4' | 'q8' | 'fp16' | 'fp32';
  readonly topK: number;
  readonly maxNewTokens: number;
  readonly defaultTemperature: number;
  readonly defaultTopP: number;
  readonly modelLoadTimeoutMs: number;
  readonly dprCap: number;
  readonly maxRetainedBranches: number;
}

const DEFAULTS: AppConfig = {
  modelId: 'HuggingFaceTB/SmolLM2-360M-Instruct',
  dtype: 'q4',
  topK: 24,
  maxNewTokens: 160,
  defaultTemperature: 0.8,
  defaultTopP: 1,
  modelLoadTimeoutMs: 180_000,
  dprCap: 1.5,
  maxRetainedBranches: 12,
};

function requireRange(name: string, value: number, min: number, max: number): number {
  if (!Number.isFinite(value) || value < min || value > max) {
    throw new ConfigError(`${name} must be a finite number in [${min}, ${max}], got ${String(value)}`);
  }
  return value;
}

export function loadConfig(overrides: Partial<AppConfig> = {}): AppConfig {
  const merged: AppConfig = { ...DEFAULTS, ...overrides };

  if (!merged.modelId || typeof merged.modelId !== 'string') {
    throw new ConfigError('modelId must be a non-empty string');
  }
  if (!['q4', 'q8', 'fp16', 'fp32'].includes(merged.dtype)) {
    throw new ConfigError(`dtype must be one of q4|q8|fp16|fp32, got ${merged.dtype}`);
  }
  requireRange('topK', merged.topK, 1, 40);
  requireRange('maxNewTokens', merged.maxNewTokens, 1, 512);
  requireRange('defaultTemperature', merged.defaultTemperature, 0, 2);
  requireRange('defaultTopP', merged.defaultTopP, 0.01, 1);
  requireRange('modelLoadTimeoutMs', merged.modelLoadTimeoutMs, 1_000, 600_000);
  requireRange('dprCap', merged.dprCap, 1, 3);
  requireRange('maxRetainedBranches', merged.maxRetainedBranches, 1, 64);

  return Object.freeze(merged);
}

export const config: AppConfig = loadConfig();
