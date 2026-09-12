import {
  AlertTriangle,
  BookOpen,
  CalendarDays,
  CheckCircle2,
  ChevronRight,
  Clock3,
  UserRound,
} from "lucide-react";

type NotebookStatus = "Review Recommended" | "Stable" | "Needs Follow-up";

type PatientNotebook = {
  id: string;
  name: string;
  age: number;
  sex: "Female" | "Male";
  mrn: string;
  lastReview: string;
  dateTime: string;
  status: NotebookStatus;
};

const patientNotebooks: readonly PatientNotebook[] = [
  {
    id: "pt-juan-dela-cruz",
    name: "Juan Dela Cruz",
    age: 72,
    sex: "Male",
    mrn: "12345678",
    lastReview: "May 19, 2025",
    dateTime: "2025-05-19",
    status: "Review Recommended",
  },
  {
    id: "pt-maria-santos",
    name: "Maria Santos",
    age: 65,
    sex: "Female",
    mrn: "87654321",
    lastReview: "May 19, 2025",
    dateTime: "2025-05-19",
    status: "Stable",
  },
  {
    id: "pt-roberto-garcia",
    name: "Roberto Garcia",
    age: 58,
    sex: "Male",
    mrn: "11223344",
    lastReview: "May 17, 2025",
    dateTime: "2025-05-17",
    status: "Needs Follow-up",
  },
  {
    id: "pt-elena-reyes",
    name: "Elena Reyes",
    age: 47,
    sex: "Female",
    mrn: "55667788",
    lastReview: "May 16, 2025",
    dateTime: "2025-05-16",
    status: "Stable",
  },
  {
    id: "pt-benjamin-lim",
    name: "Benjamin Lim",
    age: 69,
    sex: "Male",
    mrn: "99887766",
    lastReview: "May 15, 2025",
    dateTime: "2025-05-15",
    status: "Review Recommended",
  },
  {
    id: "pt-alicia-moreno",
    name: "Alicia Moreno",
    age: 61,
    sex: "Female",
    mrn: "44332211",
    lastReview: "May 14, 2025",
    dateTime: "2025-05-14",
    status: "Needs Follow-up",
  },
] as const;

function statusTone(status: NotebookStatus) {
  if (status === "Review Recommended") return "danger";
  if (status === "Needs Follow-up") return "warning";
  return "success";
}

function StatusIcon({ status }: { status: NotebookStatus }) {
  if (status === "Review Recommended") return <AlertTriangle aria-hidden="true" />;
  if (status === "Needs Follow-up") return <Clock3 aria-hidden="true" />;
  return <CheckCircle2 aria-hidden="true" />;
}

function NotebookIllustration() {
  return (
    <div className="notebook-illustration" aria-hidden="true">
      <span className="notebook-illustration__spine" />
      <span className="notebook-illustration__rings">
        {Array.from({ length: 6 }, (_, index) => <i key={index} />)}
      </span>
      <span className="notebook-illustration__avatar"><UserRound /></span>
      <span className="notebook-illustration__identity-lines"><i /><i /></span>
      <span className="notebook-illustration__copy-lines"><i /><i /><i /><i /></span>
    </div>
  );
}

type PatientNotebooksProps = {
  onOpenNotebook: (patientId: string) => void;
};

export function PatientNotebooks({ onOpenNotebook }: PatientNotebooksProps) {
  return (
    <div className="patient-notebooks">
      <h1 className="sr-only">Patient Notebooks</h1>

      <section className="notebooks-intro" aria-labelledby="notebooks-intro-title">
        <span className="notebooks-intro__icon" aria-hidden="true"><BookOpen /></span>
        <div>
          <h2 id="notebooks-intro-title">Open a notebook to view the patient overview</h2>
          <p>Each synthetic notebook contains clinical summaries, medications, alerts, and evidence.</p>
        </div>
      </section>

      <section aria-labelledby="notebook-directory-title">
        <h2 className="sr-only" id="notebook-directory-title">Synthetic patient notebook directory</h2>
        <div className="notebook-grid">
          {patientNotebooks.map((patient) => {
            const tone = statusTone(patient.status);
            return (
              <article className="notebook-card" key={patient.id}>
                <NotebookIllustration />
                <div className="notebook-card__content">
                  <h3>{patient.name}</h3>
                  <p className="notebook-card__demographics">
                    <span><UserRound aria-hidden="true" /> {patient.age}</span>
                    <span>{patient.sex}</span>
                  </p>
                  <p className="notebook-card__mrn">MRN: {patient.mrn}</p>
                  <p className="notebook-card__review-date">
                    <CalendarDays aria-hidden="true" />
                    <span>Last review: <time dateTime={patient.dateTime}>{patient.lastReview}</time></span>
                  </p>
                  <span className={`notebook-status notebook-status--${tone}`}>
                    <StatusIcon status={patient.status} />
                    {patient.status}
                  </span>
                  <button className="open-notebook-button" type="button" onClick={() => onOpenNotebook(patient.id)}>
                    <span>Open Notebook</span>
                    <ChevronRight aria-hidden="true" />
                  </button>
                </div>
              </article>
            );
          })}
        </div>
      </section>

      <p className="notebook-count">Showing {patientNotebooks.length} of {patientNotebooks.length} synthetic notebooks</p>
    </div>
  );
}
