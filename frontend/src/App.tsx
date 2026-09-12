import { useEffect, useState } from "react";

import { ClinicalDisclaimer } from "./components/ClinicalDisclaimer";
import { LandingPage } from "./components/LandingPage";
import { PatientNotebooks } from "./components/PatientNotebooks";
import { PatientOverview } from "./components/PatientOverview";
import { Sidebar } from "./components/Sidebar";
import { TopNavbar } from "./components/TopNavbar";

type ApplicationView = "landing" | "notebooks" | "overview";

function viewFromHash(): ApplicationView {
  if (window.location.hash === "#patient-notebooks") return "notebooks";
  if (window.location.hash === "#patient-overview") return "overview";
  return "landing";
}

export default function App() {
  const [view, setView] = useState<ApplicationView>(viewFromHash);
  const isOverview = view === "overview";
  const isLanding = view === "landing";

  useEffect(() => {
    const handleHashChange = () => setView(viewFromHash());
    window.addEventListener("hashchange", handleHashChange);
    return () => window.removeEventListener("hashchange", handleHashChange);
  }, []);

  const navigate = (nextView: ApplicationView) => {
    window.location.hash =
      nextView === "landing"
        ? "home"
        : nextView === "notebooks"
          ? "patient-notebooks"
          : "patient-overview";
    setView(nextView);
  };

  return (
    <div className={`app-shell app-shell--${view}`}>
      {isLanding ? (
        <LandingPage
          onOpenWorkspace={() => navigate("notebooks")}
          onViewDemo={() => navigate("overview")}
        />
      ) : (
        <>
          <TopNavbar
            pageTitle={isOverview ? "Patient Overview" : "Patient Notebooks"}
            searchPlaceholder={
              isOverview
                ? "Search patients, medications, alerts..."
                : "Search patients, notebooks, medications..."
            }
            onBrandClick={() => navigate("landing")}
          />
          <div className={`app-layout app-layout--${view}`}>
            {isOverview ? <Sidebar onPatientsSelect={() => navigate("notebooks")} /> : null}
            <main className={`main-content main-content--${view}`}>
              {isOverview ? (
                <>
                  <PatientOverview />
                  <ClinicalDisclaimer />
                </>
              ) : (
                <PatientNotebooks onOpenNotebook={() => navigate("overview")} />
              )}
            </main>
          </div>
        </>
      )}
    </div>
  );
}
