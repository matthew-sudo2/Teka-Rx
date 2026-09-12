import {
  ArrowRight,
  CheckCircle2,
  ClipboardCheck,
  Clock3,
  Plus,
  TriangleAlert,
  Users,
} from "lucide-react";

import { AsciiPill } from "./AsciiArtwork";

type MetricTone = "neutral" | "warning" | "danger" | "success";

type DashboardMetric = {
  label: string;
  value: number;
  tone: MetricTone;
  icon: typeof Users;
};

type AttentionRisk = "High" | "Moderate";

type AttentionReview = {
  patientId: string;
  medicationCount: number;
  risk: AttentionRisk;
  concern: string;
  lastReviewed: string;
  lastReviewedDateTime?: string;
};

type ReviewStatus = "Completed" | "Reviewed";

type RecentReview = {
  patientId: string;
  reviewedAt: string;
  reviewedAtDateTime: string;
  medicationCount: number;
  status: ReviewStatus;
  reviewer: string;
};

const dashboardMetrics: readonly DashboardMetric[] = [
  { label: "Active Patients", value: 12, tone: "neutral", icon: Users },
  { label: "Reviews Pending", value: 4, tone: "warning", icon: Clock3 },
  { label: "High-Risk Alerts", value: 3, tone: "danger", icon: TriangleAlert },
  { label: "Reviews Completed", value: 28, tone: "success", icon: CheckCircle2 },
];

const attentionReviews: readonly AttentionReview[] = [
  {
    patientId: "PT-1042",
    medicationCount: 7,
    risk: "High",
    concern: "Bleeding interaction",
    lastReviewed: "10:42 AM",
    lastReviewedDateTime: "10:42",
  },
  {
    patientId: "PT-1087",
    medicationCount: 9,
    risk: "High",
    concern: "CNS depression",
    lastReviewed: "9:16 AM",
    lastReviewedDateTime: "09:16",
  },
  {
    patientId: "PT-0918",
    medicationCount: 5,
    risk: "Moderate",
    concern: "Renal dosing",
    lastReviewed: "Yesterday",
  },
];

const recentReviews: readonly RecentReview[] = [
  {
    patientId: "PT-0821",
    reviewedAt: "Today, 10:41 AM",
    reviewedAtDateTime: "2026-08-27T10:41:00+08:00",
    medicationCount: 4,
    status: "Completed",
    reviewer: "Dr. Reyes",
  },
  {
    patientId: "PT-0764",
    reviewedAt: "Yesterday, 4:18 PM",
    reviewedAtDateTime: "2026-08-26T16:18:00+08:00",
    medicationCount: 6,
    status: "Reviewed",
    reviewer: "M. Johnson, PharmD",
  },
  {
    patientId: "PT-0711",
    reviewedAt: "Aug 25, 9:03 AM",
    reviewedAtDateTime: "2026-08-25T09:03:00+08:00",
    medicationCount: 3,
    status: "Completed",
    reviewer: "Dr. Reyes",
  },
];

function DashboardSummary() {
  return (
    <section className="dashboard-summary" aria-labelledby="dashboard-summary-title">
      <h2 className="sr-only" id="dashboard-summary-title">
        Medication review summary
      </h2>

      <dl className="dashboard-summary__metrics">
        {dashboardMetrics.map(({ icon: Icon, label, tone, value }) => (
          <div
            className={`dashboard-metric dashboard-metric--${tone}`}
            key={label}
          >
            <div className="dashboard-metric__icon" aria-hidden="true">
              <Icon />
            </div>
            <div className="dashboard-metric__content">
              <dt>{label}</dt>
              <dd>{value}</dd>
            </div>
          </div>
        ))}
      </dl>
    </section>
  );
}

function RequiresAttention() {
  return (
    <section
      className="dashboard-section attention-section"
      aria-labelledby="attention-title"
    >
      <header className="dashboard-section__header">
        <div>
          <p className="dashboard-section__kicker">Clinical queue</p>
          <h2 id="attention-title">Requires Attention</h2>
        </div>
        <p className="dashboard-section__summary">
          Three medication reviews need follow-up.
        </p>
      </header>

      <div className="clinical-table-wrap">
        <table className="clinical-table attention-table">
          <caption className="sr-only">
            Synthetic patient medication reviews requiring attention
          </caption>
          <thead>
            <tr>
              <th scope="col">Patient</th>
              <th scope="col">Regimen</th>
              <th scope="col">Risk</th>
              <th scope="col">Primary concern</th>
              <th scope="col">Last reviewed</th>
              <th scope="col">
                <span className="sr-only">Action</span>
              </th>
            </tr>
          </thead>
          <tbody>
            {attentionReviews.map((review) => (
              <tr key={review.patientId}>
                <th className="clinical-table__patient" scope="row">
                  {review.patientId}
                </th>
                <td>{review.medicationCount} medications</td>
                <td>
                  <span
                    className={`risk-status risk-status--${review.risk.toLowerCase()}`}
                  >
                    <span className="risk-status__dot" aria-hidden="true" />
                    {review.risk}
                  </span>
                </td>
                <td className="clinical-table__concern">{review.concern}</td>
                <td>
                  {review.lastReviewedDateTime ? (
                    <time dateTime={review.lastReviewedDateTime}>
                      {review.lastReviewed}
                    </time>
                  ) : (
                    review.lastReviewed
                  )}
                </td>
                <td className="clinical-table__action-cell">
                  <button
                    className="table-action"
                    type="button"
                    aria-label={`Review medication regimen for ${review.patientId}`}
                  >
                    <span>Review</span>
                    <ArrowRight aria-hidden="true" />
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function RecentReviews() {
  return (
    <section
      className="dashboard-section recent-reviews"
      aria-labelledby="recent-reviews-title"
    >
      <header className="dashboard-section__header">
        <div>
          <p className="dashboard-section__kicker">Review history</p>
          <h2 id="recent-reviews-title">Recent Reviews</h2>
        </div>
      </header>

      <div className="clinical-table-wrap">
        <table className="clinical-table recent-reviews__table">
          <caption className="sr-only">
            Recently completed synthetic patient medication reviews
          </caption>
          <thead>
            <tr>
              <th scope="col">Patient</th>
              <th scope="col">Reviewed</th>
              <th scope="col">Regimen</th>
              <th scope="col">Reviewer</th>
              <th scope="col">Status</th>
            </tr>
          </thead>
          <tbody>
            {recentReviews.map((review) => (
              <tr key={`${review.patientId}-${review.reviewedAtDateTime}`}>
                <th className="clinical-table__patient" scope="row">
                  {review.patientId}
                </th>
                <td>
                  <time dateTime={review.reviewedAtDateTime}>
                    {review.reviewedAt}
                  </time>
                </td>
                <td>{review.medicationCount} medications</td>
                <td>{review.reviewer}</td>
                <td>
                  <span
                    className={`review-status review-status--${review.status.toLowerCase()}`}
                  >
                    <CheckCircle2 aria-hidden="true" />
                    {review.status}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

export function DashboardWorkspace() {
  return (
    <div className="dashboard-workspace">
      <header className="dashboard-header">
        <div className="dashboard-header__copy">
          <p className="dashboard-header__context">Clinical workspace</p>
          <h1 id="dashboard-title">Medication Review Dashboard</h1>
          <p className="dashboard-header__description">
            Review patient regimens, identify medication risks, and inspect
            supporting evidence.
          </p>
        </div>

        <div className="dashboard-header__controls">
          <div className="dashboard-header__artwork" aria-hidden="true">
            <AsciiPill />
          </div>
          <button className="dashboard-primary-action" type="button">
            <Plus aria-hidden="true" />
            <ClipboardCheck aria-hidden="true" className="sr-only" />
            <span>New Medication Review</span>
          </button>
        </div>
      </header>

      <DashboardSummary />

      <div className="dashboard-workspace__sections">
        <RequiresAttention />
        <RecentReviews />
      </div>
    </div>
  );
}

