# TekaRx frontend prototype

This directory contains the static, frontend-only TekaRx prototype. It includes a public landing page, a Patient Notebooks directory, and a notebook-driven Patient Overview.

It intentionally has no backend, API calls, authentication, database, model inference, or persistent patient state. The patient records are synthetic demonstration fixtures. Hash navigation connects the two prototype views; clinical action controls remain presentational.

## Run locally

```powershell
cd frontend
npm install
npm run dev
```

Open <http://127.0.0.1:5173/>.

## Production build

```powershell
npm run build
npm run preview
```

## Visual implementation

- The shared application shell uses muted forest-green glass surfaces, neutral gray-green borders, and semantic red, amber, and green status colors.
- The landing page is the default route and is also available at `#home`.
- Patient Overview is available at `#patient-overview`; Patient Notebooks is available at `#patient-notebooks`.
- Notebook illustrations are lightweight CSS components with a restrained green spine, gray binding rings, avatar, and document lines.
- The application background combines a near-black green gradient with a fixed monochromatic SVG-turbulence grain overlay.
- The grain is embedded in CSS at very low opacity and does not require an external image asset.
- Grain is disabled for print and forced-color modes.
- Space Grotesk is bundled locally through `@fontsource`; the page does not rely on a remote font request.
