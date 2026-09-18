/** Structured logging — spec §10.2. Log events, never prose. */

export type Level = 'debug' | 'info' | 'warn' | 'error';

const DEV: boolean = typeof import.meta !== 'undefined' && Boolean(import.meta.env?.DEV);

export function log(level: Level, event: string, fields: Record<string, unknown> = {}): void {
  const line = { ts: new Date().toISOString(), level, event, ...fields };
  if (DEV) {
    const method = level === 'debug' ? 'log' : level;
    (console as unknown as Record<string, (...a: unknown[]) => void>)[method](line);
    return;
  }
  if (level === 'error' || level === 'warn') {
    console[level](JSON.stringify(line));
  }
}
