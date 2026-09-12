import { ArrowRight, CirclePlay, ClipboardCheck, ShieldCheck } from "lucide-react";

import { AsciiPill } from "./AsciiArtwork";

type HeroSectionProps = {
  onStartReview: () => void;
  onViewDemo: () => void;
};

export function HeroSection({ onStartReview, onViewDemo }: HeroSectionProps) {
  return (
    <section className="hero" id="home" aria-labelledby="hero-title">
      <div className="hero__copy">
        <p className="hero__eyebrow">
          <ShieldCheck aria-hidden="true" />
          <span>Explainable. Evidence-Linked. Research-Only.</span>
        </p>

        <h1 className="hero__title" id="hero-title">
          Safer medication
          <br />
          review, <span>made clear.</span>
        </h1>

        <p className="hero__description">
          TekaRx is a research decision-support prototype that helps clinicians review medication regimens,
          inspect safety signals, and examine the evidence behind each result.
        </p>

        <div className="hero__actions" aria-label="Demo actions">
          <button className="button button--primary" type="button" onClick={onStartReview}>
            <span>Start a Review</span>
            <ArrowRight aria-hidden="true" />
          </button>
          <button className="button button--secondary" type="button" onClick={onViewDemo}>
            <CirclePlay aria-hidden="true" />
            <span>View Demo</span>
          </button>
        </div>

        <p className="hero__assurance">
          <ShieldCheck aria-hidden="true" />
          <span>Synthetic demonstration cases. No real patient data.</span>
        </p>
      </div>

      <div className="hero__visual">
        <AsciiPill className="hero__pill" accessibleTitle="Three-dimensional capsule formed from ASCII characters representing medication-safety analysis" />
      </div>
    </section>
  );
}
