# Design Document: TekaRx Animated Dashboard

## Executive Summary

The TekaRx Animated Dashboard extends the existing Streamlit application with a carousel-based home screen, persistent navigation, and organized state management while preserving all accuracy contracts. This design emphasizes motion restraint (reduced-motion safe), slop-free copy, clinical terminology, and accessibility-first patterns. The implementation loads exact frozen model artifacts, performs RapidFuzz medication matching against the reference dictionary, displays the Model Priority Score with clinical green/orange theming, and surfaces medication mapping coverage and top feature contributors.

---

## Architecture Overview

### Application Structure

The app is organized into four primary screens accessed via persistent top navigation:

1. **Home Screen** — Animated carousel (3–4 slides) introducing TekaRx, FAERS training data, output features, and disclaimer
2. **Predict Screen** — Form entry (age, sex, optional weight, medications) with real-time matching feedback, Model Priority Score display, and guidance
3. **How It Works Screen** — Static educational content explaining the model and terminology
4. **Safety Screen** — Disclaimer, terms of use, and data handling practices

### State Management

- **Session state**: Manages form prefill, example loading, and prediction history to avoid stateless resets
- **Cached artifacts**: `@st.cache_resource` loads the frozen model bundle and drug dictionary once per app session
- **URL-based navigation**: Hash routing (`#home`, `#predict`, `#how-it-works`, `#safety`) enables browser back/forward

### Data Flow

```
User Input (form)
    ↓
Medication Parsing (_parse_medications)
    ↓
RapidFuzz Matching against drug_dictionary["faers_raw"]
    ↓
Feature Building (_build_features: age groups, ROR stats, ATC diversity, etc.)
    ↓
Imputer (train-fitted statistics) → Scaler (frozen) → predict_proba
    ↓
Model Priority Score (0–100%)
    ↓
Label Assignment (>= 50% → "Review Priority Signal" + amber; < 50% → "No Priority Signal" + green)
    ↓
Guidance Rendering + Top 5 Contributors + Summary Metrics
```

---

## Component Design

### 1. Carousel Component (Home Screen)

**Purpose**: Introduce TekaRx, build trust, and guide visitors into the Predict workflow.

**Structure**:
- 4 slides, each full-width with centered text and imagery placeholders
- **Slide 1**: "What is TekaRx?" — Headline + one-liner about medication safety evaluation
- **Slide 2**: "Trained on FAERS" — Training dataset, event count, and research-only note
- **Slide 3**: "What You Get" — Model Priority Score, medication mapping, top contributors
- **Slide 4**: "Disclaimer & Next Steps" — Terms, research use, and CTA to Predict

**Controls**:
- Left/Right arrows (positioned at slide edges, vertically centered)
- Dot indicators below slides (one per slide, clickable)
- Autoplay (with 4-second interval per slide)
- Pause on hover (visual indicator displays when paused)

**Motion**:
- Default: Fade transition (300ms) between slides
- `prefers-reduced-motion`: Instant slide change, no autoplay, arrows/dots remain functional

**Copy Applied**:
- No em dashes, no vague terms, no fabricated statistics
- Active voice, specific clinical terminology (e.g., "serious adverse event patterns", not "safety signals")
- Disclaimer on every slide: "Research decision-support, not medical advice."

### 2. Persistent Navigation Bar

**Purpose**: Enable one-click access to any screen; show active screen state.

**Structure**:
- Fixed top bar, spanning full width
- 4 nav items: Home, Predict, How It Works, Safety
- Active item highlighted in clinical green (#2FBF71) with bold font or underline
- Inactive items in secondary ink color (#485750)

**Responsive**:
- Desktop (≥ 768px): Full horizontal layout
- Mobile (< 768px): Hamburger menu or collapsible navigation

**Accessibility**:
- All nav links keyboard-navigable (Tab/Shift+Tab)
- Visible focus indicator (outline: 3px solid #b7dbca, offset: 2px)
- ARIA labels for screen reader context: `aria-current="page"` on active item
- Arrow keys for carousel navigation when carousel is focused

### 3. Predict Screen Form

**Purpose**: Collect patient demographics and medication list, then evaluate regimen.

**Layout**:
- Left column (50%): Form inputs
- Right column (50%): Empty state → Loading state → Result state

**Form Fields**:

| Field | Type | Validation | Required |
|-------|------|-----------|----------|
| Age | Number input | 0–120, integer | Yes |
| Biological Sex | Dropdown | Female / Male / Unknown | Yes |
| Weight (optional) | Number input | 0–500 kg, float | No |
| Active Medications | Text area | One per line or comma-separated | Yes |

**Button State**:
- Disabled until all required fields populated and valid
- On click: Transition to loading state
- On timeout (> 5 seconds): Show timeout message with retry button

### 4. Medication Matching & Coverage Table

**Purpose**: Show which medications were matched to the dictionary and at what confidence.

**Table Columns**:
- **Entered Medication**: User's input name
- **Mapping Status**: Matched / Ambiguous / Unmatched
- **Matched Dictionary Name**: Canonical name (if matched or ambiguous)
- **Match Confidence**: Score as percentage (e.g., "92%")
- **Suggested RapidFuzz Correction**: Top 3 alternatives if score < 90%
- **Included in Model Features**: Yes / No

**Logic**:
- Score ≥ 90%: "Matched", used=true
- Score ≥ 75% and < 90%: "Ambiguous", used=true
- Score < 75%: "Unmatched", used=false

**Warning**:
- If any unmatched medications exist: Display banner: "[N] medication(s) could not be mapped with ≥90% confidence. Please review suggestions or verify spellings."
- If all matched: No warning displayed

### 5. Model Priority Score Display

**Purpose**: Show the prediction result with clinical significance and operating context.

**Layout**:
```
┌─────────────────────────────────────┐
│ Model Priority Score                │
│ 72%                                 │
│ [Review Priority Signal - amber]    │
│                                     │
│ Operating decision threshold: 50%   │
│ Trained Random Forest paradigm      │
│                                     │
│ Meaning: [guidance text]            │
│ Next steps: [action text]           │
│ Research decision-support,          │
│ not medical advice.                 │
└─────────────────────────────────────┘
```

**Color Scheme**:
- Score font: Clinical green (#2FBF71) if < 50%; Fintech orange (#8a580a) if ≥ 50%
- Badge background: Green soft (#eaf3ed) / border: green (#c3ddcc) for "No Priority Signal"
- Badge background: Amber soft (#fef6e7) / border: amber (#f2dcab) for "Review Priority Signal"

**Typography**:
- Score: Large, prominent (4.4rem responsive max)
- Label: 0.95rem, bold
- Caption: 0.92rem secondary ink

### 6. Guidance and Recommendation Blocks

**Purpose**: Translate the Model Priority Score into actionable clinical guidance.

**Content (Score ≥ 50%)**:
- Meaning: "The model priority score is at or above the frozen operating threshold (50%). Based on the patient demographics and pharmacological features of the reported regimen, this combination exhibits a priority signal consistent with serious adverse event patterns."
- Next steps: "Conduct a clinical review of the medication list, paying particular attention to drug interactions, cumulative organ toxicities, and the top contributing factors surfaced below."

**Content (Score < 50%)**:
- Meaning: "The model priority score is below the frozen operating threshold (50%). The combination presents No Priority Signal under the trained IMRAD seriousness model."
- Next steps: "Document the current regimen and continue standard monitoring. Re-evaluate if medications are added, doses adjusted, or if new symptoms arise."

**Unmatched Note**:
- If unmatched > 0: Append: "[N] medication(s) were not matched in the clinical dictionary. Verify spellings using the suggestions below to ensure complete model coverage."

**Disclaimer**:
- Always displayed beside guidance: "Research decision-support, not medical advice."

### 7. Top Contributors Table

**Purpose**: Surface the top 5 features driving the Model Priority Score.

**Columns**:
- **Feature**: Human-readable label (e.g., "Highest Drug ROR", "Matched Medication Count", "Polypharmacy × Age Interaction")
- **Regimen Value**: Patient's specific value for that feature (e.g., "3 medications", "2.45", "74 years")
- **Relative Contribution**: Percentage of top-5 feature importance sum (e.g., "28.3%")

**Label Mapping**:
- `max_ror` → "Highest Drug ROR"
- `mean_log_ror` → "Average Log ROR"
- `high_ror_count` → "High-ROR Medication Count"
- `has_boxed_warning` → "Boxed Warning Present"
- `num_drugs` → "Matched Medication Count"
- `polypharmacy_age` → "Polypharmacy × Age Interaction"
- `atc_diversity` → "ATC Category Diversity"
- `age_imputed_years` → "Patient Age"
- Other features mapped similarly (see `_feature_label()` in existing code)

**Terminology**:
- Never use "SHAP value", "feature attribution", "explainability", or "interpretation"
- Always use "Contribution to Model Score"

**Caption**: "Top feature contributions from the frozen model pipeline for this specific patient input."

### 8. Summary Metrics Grid

**Purpose**: Provide a quick snapshot of key regimen statistics.

**Cards**:
1. **Matched Medications**: Integer count of medications with score ≥ 75%
2. **Highest Drug ROR**: Floating-point value (e.g., "3.45") from matched medications
3. **Boxed Warning**: "Present" or "None"

**Layout**:
- Desktop (≥ 768px): 3-column grid
- Mobile (< 768px): Stack vertically

**Styling**: Neutral border, light background, clear typography

---

## State Machine

### Screen States

```
┌─────────────────────────────────────────────────┐
│            Artifact Check                       │
└──────────────┬──────────────────────────────────┘
               │
        ┌──────┴──────┐
        │             │
     Missing        Present
        │             │
        ▼             ▼
   [Error State]  [Home Screen]
                      │
              ┌───────┴───────┐
              │               │
         [Predict Screen] [Other Screens]
              │
        ┌─────┴──────────────┐
        │                    │
    [Empty State]      [Form with data]
        │                    │
        └────────┬───────────┘
                 │
            [Loading State]
                 │
        ┌────────┴────────┐
        │                 │
    [Success State]  [Error State]
```

### Transitions

- **Empty → Loading**: Click "Run TekaRx Score" with valid form
- **Loading → Success**: Model evaluation completes (typical < 1 second)
- **Loading → Timeout**: No response after 5 seconds; show retry button
- **Success → Empty**: Click "Reset" or navigate away; clear session state
- **Error**: Artifact missing at load or validation failure; display recovery panel

---

## Styling and Color System

### Color Palette

| Role | Color | Hex | WCAG AA Contrast |
|------|-------|-----|------------------|
| Primary text (ink) | Deep ink | #16221c | 21:1 on white |
| Secondary text | Ink secondary | #485750 | 8:1 on white |
| Background | White | #ffffff | - |
| Light panel background | Near-white | #f8faf9 | - |
| Borders | Neutral grey | #d8e2dc | 3:1 on white |
| Success/Active | Clinical green | #2FBF71 | 4.2:1 on white |
| Success background | Green soft | #eaf3ed | - |
| Success border | Green border | #c3ddcc | - |
| Warning/Secondary | Fintech orange | #8a580a | 7:1 on white |
| Warning background | Amber soft | #fef6e7 | - |
| Warning border | Amber border | #f2dcab | - |
| Error/High priority | Red primary | #9b2828 | 6:1 on white |
| Error background | Red soft | #fbeeed | - |

### Typography

- **Font family**: System stack (`-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif`)
- **Headings** (H1–H3): Bold, deep ink, no em dashes
- **Body text**: 0.95–1rem, deep ink, line-height 1.5
- **Captions**: 0.88–0.92rem, secondary ink
- **Monospace** (code, commands): Same system monospace, for error panels and examples

### Components

- **Button**: 6px border-radius, 2.85rem min-height, green bg with white text, 0.15s transition
- **Form input**: 1px border (neutral grey), 6px radius, padding 0.75rem
- **Cards**: 7px radius, 1px border, light panel background, 0.75–0.85rem padding
- **Disclaimer**: Left border (3px green), left padding 0.65rem, 0.88rem text

---

## Accessibility Design

### Keyboard Navigation

1. **Tab order** (logical reading order):
   - Navigation bar → Form inputs → Submit button → Results (if present)
2. **Carousel keys**:
   - Left/Right arrows: Navigate slides
   - Home: Jump to slide 1
   - End: Jump to last slide
   - 1–4 number keys: Jump to specific slide (if present)
3. **Focus indicators**: 3px solid outline (#b7dbca) with 2px offset, visible on all interactive elements

### Screen Reader Support

- **Navigation**: Active item marked with `aria-current="page"`
- **Form labels**: Explicit `<label for="...">` associations
- **Error messages**: `aria-describedby` linking field to error text
- **Loading state**: `role="status"` with `aria-live="polite"` for real-time updates
- **Result section**: `role="region"` with `aria-label="Medication priority score results"`
- **Disclaimer**: Explicitly announced as "Research decision-support" on every slide and result

### Color Contrast

- All text on backgrounds: Minimum 4.5:1 (WCAG AA)
- Large text (18px+): Minimum 3:1
- Verifiable against entire gradient or background area (not point-sample)

### Reduced Motion

Applied via `@media (prefers-reduced-motion: reduce)`:
- Carousel: Disable autoplay, instant slide change (no transition)
- Loading spinner: Replace with static text + icon
- All transitions and animations: `transition: none`
- Focus outlines: Remain visible (non-animated)

---

## Error Handling

### Missing Artifacts

**Detection**: At app load, check for:
- `data/processed/models/imrad_models.joblib`
- `data/processed/drug_dictionary.parquet`

**Display**:
```
┌──────────────────────────────────────┐
│ ⚠ Model artifacts are not ready yet  │
│                                      │
│ Missing files:                       │
│ - data/processed/models/...joblib    │
│ - data/processed/drug_dictionary...  │
│                                      │
│ Build from repository root:          │
│ python -m pip install -e .           │
│ [imrad,notebook]                     │
│ python build_imrad_artifacts.py      │
│                                      │
│ Research decision-support,           │
│ not medical advice.                  │
└──────────────────────────────────────┘
```

### Validation Errors

**Age field**:
- If age < 0 or > 120: Show inline error "Age must be 0–120 years"

**Medications field**:
- If empty and form submitted: Show error "Enter at least one medication"

**RapidFuzz Matching**:
- If all medications unmatched: Show warning banner with count and suggestion links

### Timeout Handling

- If prediction takes > 5 seconds: Display "Model evaluation is taking longer than expected" with "Retry" button
- Retry: Reset loading state and re-submit form

---

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Carousel Navigation is Reversible

For any carousel with N slides and current slide index I, clicking the left arrow should advance to slide (I - 1) mod N, and clicking the right arrow should advance to slide (I + 1) mod N. Clicking a dot indicator with index J should navigate directly to slide J.

**Validates: Requirements 1.3, 1.4, 1.5, 1.9**

### Property 2: Active Navigation State Matches Screen Context

For any screen (Home, Predict, How It Works, Safety), the persistent navigation bar SHALL display exactly one nav item with the active visual state (clinical green color and bold font), and that item SHALL correspond to the current screen.

**Validates: Requirements 3.2, 3.3, 3.4, 3.5, 3.6**

### Property 3: Model Priority Score Label is Determined by Threshold

For any prediction probability P in [0.0, 1.0] and threshold T = 0.50, if P >= T, the label SHALL be "Review Priority Signal" with amber badge; if P < T, the label SHALL be "No Priority Signal" with green badge.

**Validates: Requirements 7.2, 7.3, 7.6**

### Property 4: Medication Matching Classification is Consistent

For any medication name matched against the dictionary with score S in [0.0, 100.0], if S >= 90, the status SHALL be "Matched" and used SHALL be true; if 75 <= S < 90, the status SHALL be "Ambiguous" and used SHALL be true; if S < 75, the status SHALL be "Unmatched" and used SHALL be false.

**Validates: Requirements 5.2, 5.3, 5.4**

### Property 5: Unmatched Medication Warning Presence is Correct

For any medication matching result set, if the count of unmatched medications > 0, a warning banner SHALL be displayed with the exact count; if the count = 0, no warning SHALL be displayed.

**Validates: Requirements 5.6, 5.7**

### Property 6: Feature Importance Contributions Sum Correctly

For any top-5 feature table displayed, the sum of all "Relative Contribution" percentages SHALL equal 100% (within rounding error of ±1%).

**Validates: Requirements 9.6**

### Property 7: Reduced Motion Disables Carousel Autoplay

For any system with `prefers-reduced-motion: reduce` set and the carousel displayed, autoplay SHALL be disabled and slides SHALL change instantly without transition animation.

**Validates: Requirements 1.8, 16.2, 16.3**

### Property 8: Form Submission is Blocked on Invalid Input

For any form with age field value < 0 or > 120, or with all required fields empty (age, sex, medications), the "Run TekaRx Score" button SHALL remain disabled and form submission SHALL be prevented.

**Validates: Requirements 4.6, 4.8**

### Property 9: Model Prediction Outputs Match Artifact Execution

For any patient input and set of matched medications, running the app prediction pipeline (imputer → scaler → predict_proba) SHALL produce a probability output identical to the frozen model bundle execution as documented in `docs/streamlit_dashboard.md` Python reproduction example.

**Validates: Requirements 20.2, 20.3, 20.4, 20.5, 20.9**

### Property 10: Copy is Free of Marketing Slop

For any text rendered in the app (carousel slides, form labels, buttons, guidance, disclaimers), the text SHALL NOT contain em dash characters (`—`), and SHALL NOT use pronouns in place of specific referents (e.g., "THE App" not "It").

**Validates: Requirements 14.1, 14.5**

### Property 11: Accessibility Focus Indicators are Always Visible

For any interactive element (button, link, form input, carousel arrow) that receives keyboard focus, a visible focus indicator SHALL be displayed with contrast ratio >= 4.5:1 (WCAG AA).

**Validates: Requirements 15.2, 15.7**

### Property 12: Keyboard Tab Navigation is Consistent

For any sequence of Tab key presses from the app start, the focus order SHALL follow logical visual order (top-to-bottom, left-to-right), and pressing Tab from the last focusable element SHALL cycle back to the first focusable element.

**Validates: Requirements 15.1, 15.9**

### Property 13: Sticky Navigation Persists Across Screen Transitions

For any navigation between screens (Home, Predict, How It Works, Safety), the persistent navigation bar SHALL remain visible at the top of the viewport and SHALL be fully functional regardless of which screen is displayed.

**Validates: Requirements 3.1**

### Property 14: Feature Defaults Preserve Train-Fitted Imputer Statistics

For any feature column that cannot be derived from user input (e.g., dosage, patient graph metrics), the feature value SHALL be initialized from the train-fitted imputer statistics stored in the frozen model bundle.

**Validates: Requirements 20.3**

### Property 15: All Results Include the Research Disclaimer

For any successful prediction result displayed, the one-line disclaimer "Research decision-support, not medical advice." SHALL appear beside the Model Priority Score, guidance, and contributors sections.

**Validates: Requirements 7.5, 8.6**

### Property 16: RapidFuzz Matching Uses Exact Scorer and Source

For any medication matching operation, the fuzzy matching logic SHALL use RapidFuzz WRatio scorer against the `drug_dictionary.parquet["faers_raw"]` column, and SHALL NOT use alternative scorers or dictionary columns.

**Validates: Requirements 5.1, 20.6**

### Property 17: Modal/Dialog States Block Background Interaction

For any error state or loading state displayed (artifact error panel, timeout message), the background form or page content SHALL be non-interactive (disabled or hidden) and focus SHALL be trapped within the modal or error region.

**Validates: Requirements 11.6**

### Property 18: Carousel Slides Display Correct Content

For carousel slide 1, the content SHALL introduce TekaRx and medication safety. For slide 2, content SHALL mention FAERS training and research-only scope. For slide 3, content SHALL list Model Priority Score, medication mapping, and top contributors. For slide 4 (if present), content SHALL display disclaimer, terms, and CTA to Predict.

**Validates: Requirements 2.1, 2.2, 2.3, 2.4**

### Property 19: Empty State Displays Until First Prediction

For any first visit to the Predict Screen or after form reset, the empty-state panel SHALL be displayed showing "No Regimen Evaluated Yet", two example buttons, and guidance text. After the first successful prediction, the empty state SHALL be replaced by the result state on subsequent form submissions.

**Validates: Requirements 12.1, 12.3, 12.6**

### Property 20: Summary Metrics Grid Contains Exactly Three Cards

For any result display, the summary metrics grid SHALL contain exactly three cards: "Matched Medications", "Highest Drug ROR", and "Boxed Warning", and no additional or missing cards.

**Validates: Requirements 10.1, 10.2, 10.3, 10.4**

---

## Implementation Notes

### Extending the Existing App

The design builds on `frontend/streamlit_app.py` by:
1. Retaining all existing functions (`_artifact_status()`, `_load_artifacts()`, `_match_medications()`, `_predict()`, etc.)
2. Adding new state-management functions for carousel and navigation
3. Refactoring the main layout to support multi-screen navigation and the carousel
4. Preserving the CSS color palette and extending it with new carousel-specific styles

### Key Functions (New)

- `_render_carousel()` — Render carousel with slides, arrows, dots, and autoplay logic
- `_render_navigation()` — Render persistent top nav with active state tracking
- `_handle_carousel_input()` — Process arrow key presses, dot clicks, and number key navigation
- `_apply_reduced_motion()` — Detect system preference and disable animations

### Performance Considerations

- Carousel image preloading: Not needed (text-based slides)
- Model artifact caching: `@st.cache_resource` ensures single load per session
- Medication matching: RapidFuzz is fast (<100ms for typical regimens); no additional optimization needed

### Browser Compatibility

- Modern browsers (Chrome, Firefox, Safari, Edge) supporting ES6, CSS Grid, and `prefers-reduced-motion`
- Streamlit's built-in handling of form state and session management ensures consistency

---

## Delivery Acceptance Criteria

All work is complete when:

1. ✅ Carousel renders all 4 slides with smooth navigation (arrow keys, dot clicks, autoplay with pause-on-hover)
2. ✅ Persistent navigation shows active states and allows one-click access to all 4 screens
3. ✅ Predict screen loads with form (age, sex, weight, medications) and empty state with example buttons
4. ✅ Medication matching displays coverage table with Matched/Ambiguous/Unmatched classifications
5. ✅ Model Priority Score displays with correct label and color (green for < 50%, amber for >= 50%)
6. ✅ Guidance text and next steps are rendered based on score threshold
7. ✅ Top 5 contributors table shows features, values, and relative contribution percentages
8. ✅ Summary metrics grid displays matched count, highest ROR, boxed warning status
9. ✅ All UI states exercise successfully with real model artifacts (empty, loading, success, error)
10. ✅ Copy passes antislop review (no em dashes, no vague terms, no fabricated claims)
11. ✅ Reduced-motion support fully functional (no animations, static loading indicator, autoplay disabled)
12. ✅ Keyboard navigation and focus indicators meet WCAG AA requirements
13. ✅ Contrast ratios verified for all text on backgrounds (>= 4.5:1)
14. ✅ Missing artifact error panel displays with exact recovery commands
15. ✅ README updated with run steps and antislop mention
16. ✅ Code passes `ruff check` and `ruff format` with no errors
17. ✅ All model accuracy contracts preserved (frozen threshold, feature column order, exact artifact loading, RapidFuzz matching, no retraining)
