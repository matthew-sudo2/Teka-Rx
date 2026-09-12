import { Bell, ChevronDown, Search, ShieldPlus } from "lucide-react";

type TopNavbarProps = {
  pageTitle: string;
  searchPlaceholder: string;
  onBrandClick?: () => void;
};

export function TopNavbar({ pageTitle, searchPlaceholder, onBrandClick }: TopNavbarProps) {
  return (
    <header className="top-navbar">
      <button className="brand" type="button" onClick={onBrandClick} aria-label="Go to the TekaRx landing page">
        <span className="brand__mark" aria-hidden="true">
          <ShieldPlus strokeWidth={1.7} />
        </span>
        <span className="brand__name">TekaRx</span>
      </button>

      <p className="top-navbar__page-title">{pageTitle}</p>

      <label className="top-navbar__search">
        <span className="sr-only">Search TekaRx</span>
        <Search aria-hidden="true" />
        <input type="search" placeholder={searchPlaceholder} />
      </label>

      <div className="top-navbar__tools">
        <button className="notification-control" type="button" aria-label="Notifications">
          <Bell aria-hidden="true" />
          <span className="notification-control__dot" aria-hidden="true" />
        </button>

        <button className="profile-control" type="button" aria-label="Open Dr. Reyes profile menu">
          <span className="profile-control__avatar" aria-hidden="true">DR</span>
          <span className="profile-control__name">Dr. Reyes</span>
          <ChevronDown aria-hidden="true" />
        </button>
      </div>
    </header>
  );
}
