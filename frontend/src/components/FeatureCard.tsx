import type { ReactElement } from "react";
import { ArrowRight } from "lucide-react";

type FeatureCardProps = {
  title: string;
  description: string;
  footer: string;
  icon: ReactElement;
  accent?: "green" | "risk";
};

export function FeatureCard({
  title,
  description,
  footer,
  icon,
  accent = "green",
}: FeatureCardProps) {
  return (
    <article className={`feature-card feature-card--${accent}`}>
      <div className="feature-card__icon-wrap" aria-hidden="true">
        {icon}
      </div>
      <div className="feature-card__copy">
        <h2>{title}</h2>
        <p>{description}</p>
      </div>

      <div className="feature-card__footer">
        <span>{footer}</span>
        <ArrowRight aria-hidden="true" />
      </div>
    </article>
  );
}
