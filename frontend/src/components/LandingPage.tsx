import { ArrowRight, ShieldPlus } from "lucide-react";

import { AsciiBrain, AsciiDocument, AsciiWarning } from "./AsciiArtwork";
import { ClinicalDisclaimer } from "./ClinicalDisclaimer";
import { FeatureCard } from "./FeatureCard";
import { HeroSection } from "./HeroSection";

type LandingPageProps = {
  onOpenWorkspace: () => void;
  onViewDemo: () => void;
};

const features = [
  {
    title: "Risk Assessment",
    description: "Identify medication-related factors associated with serious adverse-event reports.",
    footer: "Prioritize review",
    icon: <AsciiWarning />,
    accent: "risk" as const,
  },
  {
    title: "Explainable Analysis",
    description: "Inspect patient features and medication contributions behind each model result.",
    footer: "Understand the signal",
    icon: <AsciiBrain />,
    accent: "green" as const,
  },
  {
    title: "Evidence Review",
    description: "Connect alerts to versioned labels, statistical signals, and supporting sources.",
    footer: "Review the evidence",
    icon: <AsciiDocument />,
    accent: "green" as const,
  },
] as const;

export function LandingPage({ onOpenWorkspace, onViewDemo }: LandingPageProps) {
  return (
    <div className="landing-page">
      <header className="landing-header">
        <a className="brand" href="#home" aria-label="TekaRx home">
          <span className="brand__mark" aria-hidden="true">
            <ShieldPlus strokeWidth={1.7} />
          </span>
          <span className="brand__name">TekaRx</span>
        </a>

        <p className="landing-header__tagline">Teka muna — review medication safety</p>

        <nav className="landing-nav" aria-label="Landing page navigation">
          <a href="#home">Home</a>
          <a href="#features">Features</a>
          <a href="#about">About</a>
        </nav>

        <button className="landing-header__workspace" type="button" onClick={onOpenWorkspace}>
          <span>Open Workspace</span>
          <ArrowRight aria-hidden="true" />
        </button>
      </header>

      <main className="landing-main">
        <HeroSection onStartReview={onOpenWorkspace} onViewDemo={onViewDemo} />

        <section className="landing-features" id="features" aria-label="TekaRx capabilities">
          {features.map((feature) => (
            <FeatureCard key={feature.title} {...feature} />
          ))}
        </section>

        <ClinicalDisclaimer />
      </main>
    </div>
  );
}
