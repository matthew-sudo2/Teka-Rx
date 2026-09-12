import {
  AlertTriangle,
  ArrowRight,
  BookOpen,
  CheckCircle2,
  ChevronRight,
  ClipboardList,
  History,
  Pill,
  Search,
  ShieldCheck,
  UserRound,
} from "lucide-react";

const conditions = [
  "Atrial Fibrillation",
  "Hypertension",
  "Type 2 Diabetes",
  "CKD Stage 3a",
  "Osteoarthritis",
] as const;

const medications = [
  { name: "Warfarin", dose: "5 mg daily", level: "critical", width: 92 },
  { name: "Aspirin", dose: "81 mg daily", level: "critical", width: 71 },
  { name: "Ibuprofen", dose: "400 mg TID PRN", level: "critical", width: 49 },
  { name: "Metformin", dose: "500 mg BID", level: "warning", width: 38 },
  { name: "Amlodipine", dose: "5 mg daily", level: "warning", width: 29 },
  { name: "Atorvastatin", dose: "20 mg nightly", level: "warning", width: 23 },
  { name: "Omeprazole", dose: "20 mg daily", level: "acceptable", width: 17 },
] as const;

const alerts = [
  {
    title: "Drug–Label Warning",
    description: "Warfarin + Aspirin: increased risk of bleeding.",
    severity: "High",
    tone: "danger",
    date: "May 19, 2025",
    time: "10:42 AM",
    dateTime: "2025-05-19T10:42:00",
  },
  {
    title: "Known Interaction",
    description: "Warfarin + Ibuprofen: bleeding risk may increase.",
    severity: "Moderate",
    tone: "warning",
    date: "May 18, 2025",
    time: "2:11 PM",
    dateTime: "2025-05-18T14:11:00",
  },
  {
    title: "Review Completed",
    description: "Clinical review completed by Dr. Reyes.",
    severity: "Completed",
    tone: "success",
    date: "May 15, 2025",
    time: "9:16 AM",
    dateTime: "2025-05-15T09:16:00",
  },
] as const;

type PanelHeadingProps = {
  icon: typeof UserRound;
  title: string;
};

function PanelHeading({ icon: Icon, title }: PanelHeadingProps) {
  return (
    <header className="panel-heading">
      <Icon aria-hidden="true" />
      <h2>{title}</h2>
    </header>
  );
}

function PatientSummary() {
  return (
    <section className="clinical-panel patient-summary" aria-labelledby="patient-summary-title">
      <header className="panel-heading">
        <UserRound aria-hidden="true" />
        <h2 id="patient-summary-title">Patient Summary</h2>
      </header>

      <div className="patient-summary__identity">
        <div className="patient-avatar" aria-hidden="true">
          <UserRound />
        </div>
        <div>
          <h3>Juan Dela Cruz</h3>
          <p>
            <span>72</span>
            <span>Male</span>
            <span>MRN: 12345678</span>
          </p>
        </div>
      </div>

      <div className="patient-summary__section">
        <h3>Conditions</h3>
        <ul className="condition-list" aria-label="Patient conditions">
          {conditions.map((condition) => (
            <li key={condition}>{condition}</li>
          ))}
        </ul>
      </div>

      <div className="patient-summary__section patient-summary__medications">
        <h3>Current Medications ({medications.length})</h3>
        <ul className="medication-text-list">
          {medications.map((medication) => (
            <li key={medication.name}>
              <span>{medication.name}</span>
              <span>{medication.dose}</span>
            </li>
          ))}
        </ul>
      </div>

      <button className="secondary-action patient-summary__profile" type="button">
        <span>View Full Profile</span>
        <ChevronRight aria-hidden="true" />
      </button>
    </section>
  );
}

function RiskSnapshot() {
  return (
    <section className="clinical-panel risk-snapshot" aria-labelledby="risk-snapshot-title">
      <header className="panel-heading">
        <ShieldCheck aria-hidden="true" />
        <h2 id="risk-snapshot-title">Risk Snapshot</h2>
      </header>

      <div className="risk-snapshot__body">
        <div className="risk-gauge" role="img" aria-label="Risk score 0.82 out of 1">
          <div className="risk-gauge__inner">
            <strong>0.82</strong>
            <span>Risk Score</span>
            <small>(0–1.0)</small>
          </div>
        </div>

        <div className="risk-snapshot__status">
          <div className="review-recommended">
            <AlertTriangle aria-hidden="true" />
            <span>Review Recommended</span>
          </div>
          <p>Model Confidence</p>
          <strong><ShieldCheck aria-hidden="true" /> High</strong>
          <span>0.87</span>
        </div>
      </div>

      <p className="risk-snapshot__note">
        This synthetic case has multiple medication-related factors associated with recorded adverse-event risk.
      </p>
    </section>
  );
}

function NextSteps() {
  const steps = [
    {
      title: "Review Medication List",
      description: "Assess current medications and interactions",
      icon: Search,
      primary: false,
    },
    {
      title: "View Evidence",
      description: "See labels and supporting evidence",
      icon: BookOpen,
      primary: false,
    },
    {
      title: "Start Clinical Review",
      description: "Begin comprehensive medication review",
      icon: ClipboardList,
      primary: true,
    },
  ] as const;

  return (
    <section className="clinical-panel next-steps" aria-labelledby="next-steps-title">
      <PanelHeading icon={ClipboardList} title="Next Steps" />
      <div className="next-steps__list">
        {steps.map(({ description, icon: Icon, primary, title }) => (
          <button className={`next-step${primary ? " next-step--primary" : ""}`} key={title} type="button">
            <Icon aria-hidden="true" />
            <span>
              <strong>{title}</strong>
              <small>{description}</small>
            </span>
            <ChevronRight aria-hidden="true" />
          </button>
        ))}
      </div>
    </section>
  );
}

function RecentAlerts() {
  return (
    <section className="clinical-panel recent-alerts" aria-labelledby="recent-alerts-title">
      <PanelHeading icon={History} title="Recent Alerts & Reviews" />
      <div className="recent-alerts__list">
        {alerts.map((alert) => (
          <article className="alert-row" key={alert.title}>
            <span className={`alert-row__icon alert-row__icon--${alert.tone}`} aria-hidden="true">
              {alert.tone === "success" ? <CheckCircle2 /> : <AlertTriangle />}
            </span>
            <div className="alert-row__copy">
              <h3>{alert.title}</h3>
              <p>{alert.description}</p>
            </div>
            <span className={`status-badge status-badge--${alert.tone}`}>{alert.severity}</span>
            <time dateTime={alert.dateTime}>
              <span>{alert.date}</span>
              <span>{alert.time}</span>
            </time>
          </article>
        ))}
      </div>
      <button className="text-action" type="button">
        <span>View All History</span>
        <ArrowRight aria-hidden="true" />
      </button>
    </section>
  );
}

function CurrentMedications() {
  return (
    <section className="clinical-panel current-medications" aria-labelledby="current-medications-title">
      <PanelHeading icon={Pill} title={`Current Medications (${medications.length})`} />
      <ul className="medication-bars">
        {medications.map((medication) => (
          <li key={medication.name}>
            <span className="medication-bars__name">{medication.name}</span>
            <span className="medication-bars__track" aria-hidden="true">
              <span
                className={`medication-bars__fill medication-bars__fill--${medication.level}`}
                style={{ width: `${medication.width}%` }}
              />
            </span>
            <span className="medication-bars__dose">{medication.dose}</span>
          </li>
        ))}
      </ul>
      <p className="current-medications__note">
        <ShieldCheck aria-hidden="true" />
        <span>Medication bars summarize review priority for this synthetic demonstration case.</span>
      </p>
    </section>
  );
}

export function PatientOverview() {
  return (
    <div className="patient-overview">
      <h1 className="sr-only">Patient Overview</h1>
      <div className="patient-overview__grid">
        <PatientSummary />
        <RiskSnapshot />
        <NextSteps />
        <RecentAlerts />
        <CurrentMedications />
      </div>
    </div>
  );
}
