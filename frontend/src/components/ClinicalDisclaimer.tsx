import { ShieldCheck } from "lucide-react";

export function ClinicalDisclaimer() {
  return (
    <aside className="clinical-disclaimer" id="about" aria-labelledby="clinical-disclaimer-title">
      <span className="clinical-disclaimer__mark" aria-hidden="true">
        <ShieldCheck />
      </span>
      <div className="clinical-disclaimer__copy">
        <p id="clinical-disclaimer-title">
          TekaRx supports clinical decision-making. It does not replace professional judgment.
        </p>
        <p>Always use clinical judgment. Not a substitute for professional advice.</p>
      </div>
    </aside>
  );
}
