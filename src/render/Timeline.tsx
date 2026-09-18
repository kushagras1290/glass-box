/**
 * Token timeline — spec §8.1. DOM, not canvas: this text must be selectable,
 * copyable and reachable by assistive tech. Each token is a real <button>.
 */
import { useEffect, useRef, useState } from 'react';
import type { BranchNode } from '../state/branchStore';

interface Props {
  readonly nodes: BranchNode[];
  readonly onFork: (step: number, tokenId: number) => void;
  readonly disabled: boolean;
}

function probabilityClass(p: number): string {
  if (p >= 0.6) return 'tok-confident';
  if (p >= 0.25) return 'tok-mid';
  return 'tok-precarious';
}

export function Timeline({ nodes, onFork, disabled }: Props) {
  const [openStep, setOpenStep] = useState<number | null>(null);
  const scrollerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = scrollerRef.current;
    if (el) el.scrollLeft = el.scrollWidth;
  }, [nodes.length]);

  const open = nodes.find((n) => n.step === openStep) ?? null;

  return (
    <div className="timeline">
      <div className="timeline-track" ref={scrollerRef} role="list" aria-label="Generated tokens">
        {nodes.map((node) => (
          <button
            key={node.id}
            type="button"
            role="listitem"
            className={`tok ${probabilityClass(node.p)} ${openStep === node.step ? 'is-open' : ''} ${
              node.chosenByUser ? 'is-forced' : ''
            }`}
            aria-expanded={openStep === node.step}
            aria-label={`Token ${node.step + 1}: ${node.text}, ${Math.round(node.p * 100)} percent${
              node.chosenByUser ? ', chosen by you' : ''
            }. Activate to see alternatives.`}
            onClick={() => setOpenStep(openStep === node.step ? null : node.step)}
            disabled={disabled}
          >
            <span className="tok-text">{node.text}</span>
            <span className="tok-bar" aria-hidden="true">
              <i style={{ width: `${Math.max(2, node.p * 100)}%` }} />
            </span>
          </button>
        ))}
      </div>

      {open && (
        <div className="alternatives" role="dialog" aria-label={`Alternatives for token ${open.step + 1}`}>
          <p className="alternatives-head">
            instead of <strong>{open.text}</strong> · pick a different future
          </p>
          <ul role="listbox" aria-label="Alternative tokens">
            {open.candidates.slice(0, 8).map((candidate) => (
              <li key={candidate.tokenId}>
                <button
                  type="button"
                  role="option"
                  aria-selected={candidate.tokenId === open.tokenId}
                  className={candidate.tokenId === open.tokenId ? 'alt is-current' : 'alt'}
                  onClick={() => {
                    setOpenStep(null);
                    onFork(open.step, candidate.tokenId);
                  }}
                  disabled={disabled}
                >
                  <span className="alt-text">{candidate.text}</span>
                  <span className="alt-bar" aria-hidden="true">
                    <i style={{ width: `${Math.max(2, candidate.p * 100)}%` }} />
                  </span>
                  <span className="alt-pct">{(candidate.p * 100).toFixed(1)}%</span>
                </button>
              </li>
            ))}
          </ul>
          <button type="button" className="alt-close" onClick={() => setOpenStep(null)}>
            close
          </button>
        </div>
      )}
    </div>
  );
}
