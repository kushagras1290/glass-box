/** Exception hierarchy — spec §10.1. No bare `throw new Error` anywhere in src/. */

export type ErrorCode =
  | 'UNSUPPORTED_DEVICE'
  | 'MODEL_LOAD'
  | 'MODEL_TIMEOUT'
  | 'OUT_OF_MEMORY'
  | 'TOKENIZE'
  | 'INFERENCE'
  | 'PROTOCOL'
  | 'CONFIG';

export class GlassBoxError extends Error {
  readonly code: ErrorCode;
  readonly recoverable: boolean;
  override readonly cause?: unknown;

  constructor(message: string, code: ErrorCode, recoverable: boolean, cause?: unknown) {
    super(message);
    this.name = new.target.name;
    this.code = code;
    this.recoverable = recoverable;
    this.cause = cause;
  }
}

export class UnsupportedDeviceError extends GlassBoxError {
  constructor(message: string, cause?: unknown) {
    super(message, 'UNSUPPORTED_DEVICE', false, cause);
  }
}

export class ModelLoadError extends GlassBoxError {
  constructor(message: string, cause?: unknown) {
    super(message, 'MODEL_LOAD', true, cause);
  }
}

export class ModelTimeoutError extends GlassBoxError {
  constructor(message: string, cause?: unknown) {
    super(message, 'MODEL_TIMEOUT', true, cause);
  }
}

export class OutOfMemoryError extends GlassBoxError {
  constructor(message: string, cause?: unknown) {
    super(message, 'OUT_OF_MEMORY', false, cause);
  }
}

export class TokenizeError extends GlassBoxError {
  constructor(message: string, cause?: unknown) {
    super(message, 'TOKENIZE', true, cause);
  }
}

export class InferenceError extends GlassBoxError {
  constructor(message: string, cause?: unknown) {
    super(message, 'INFERENCE', true, cause);
  }
}

export class ProtocolError extends GlassBoxError {
  constructor(message: string, cause?: unknown) {
    super(message, 'PROTOCOL', false, cause);
  }
}

export class ConfigError extends GlassBoxError {
  constructor(message: string, cause?: unknown) {
    super(message, 'CONFIG', false, cause);
  }
}

/** Classify an unknown thrown value into our hierarchy without string-matching at call sites. */
export function toGlassBoxError(err: unknown, fallback: ErrorCode = 'INFERENCE'): GlassBoxError {
  if (err instanceof GlassBoxError) return err;
  const message = err instanceof Error ? err.message : String(err);
  if (/out of memory|allocation failed|oom/i.test(message)) return new OutOfMemoryError(message, err);
  if (/webgpu|adapter|gpu device/i.test(message)) return new UnsupportedDeviceError(message, err);
  return new GlassBoxError(message, fallback, true, err);
}
