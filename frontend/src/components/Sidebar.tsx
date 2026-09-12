import {
  BellRing,
  BookOpen,
  ClipboardList,
  PanelLeftClose,
  Pill,
  Settings,
  UsersRound,
  type LucideIcon,
} from "lucide-react";

type SidebarItem = {
  label: string;
  icon: LucideIcon;
  active?: boolean;
  badge?: number;
};

const sidebarItems: SidebarItem[] = [
  { label: "Patients", icon: UsersRound, active: true },
  { label: "Medications", icon: Pill },
  { label: "Alerts", icon: BellRing, badge: 6 },
  { label: "Evidence", icon: BookOpen },
  { label: "Review Logs", icon: ClipboardList },
  { label: "Settings", icon: Settings },
];

type SidebarProps = {
  onPatientsSelect: () => void;
};

export function Sidebar({ onPatientsSelect }: SidebarProps) {
  return (
    <aside className="sidebar" aria-label="Application navigation">
      <nav className="sidebar__navigation">
        <ul className="sidebar__items">
          {sidebarItems.map((item) => {
            const Icon = item.icon;

            return (
              <li key={item.label}>
                <button
                  className={`sidebar__link${item.active ? " sidebar__link--active" : ""}`}
                  type="button"
                  onClick={item.label === "Patients" ? onPatientsSelect : undefined}
                  aria-current={item.active ? "page" : undefined}
                >
                  <Icon className="sidebar__icon" strokeWidth={1.6} aria-hidden="true" />
                  <span>{item.label}</span>
                  {item.badge !== undefined ? (
                    <span className="sidebar__badge" aria-label={`${item.badge} alerts`}>
                      {item.badge}
                    </span>
                  ) : null}
                </button>
              </li>
            );
          })}
        </ul>
      </nav>

      <button className="sidebar__collapse" type="button" aria-label="Collapse navigation">
        <PanelLeftClose aria-hidden="true" />
        <span>Collapse</span>
      </button>
    </aside>
  );
}
