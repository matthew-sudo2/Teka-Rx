# Requirements Document: TekaRx Animated Dashboard

## Introduction

The TekaRx animated dashboard extends the existing `frontend/streamlit_app.py` with a carousel-based home screen, persistent navigation, and an enhanced predict interface. The feature preserves all existing accuracy contracts (frozen models, artifact loading, RapidFuzz medication matching, feature column order) while introducing motion, state management, and organized navigation. The dashboard maintains slop-free copy per `antislop.md`, applies clinical terminology from the capstone draft (Model Priority Score, No Priority Signal, Medication Mapping Coverage), and implements accessibility patterns including reduced-motion safety, keyboard navigation, and focus management.

## Glossary

- **App**: The Streamlit application extending `frontend/streamlit_app.py`
- **Home Screen**: The entry point, containing an animated carousel with 3–4 slides
- **Carousel**: A slide viewer with left/right arrow navigation, dot indicators, and pause-on-hover
- **Navigation Bar**: Persistent top navigation showing Home, Predict, How It Works, and Safety with active states
- **Predict Screen**: The medication evaluation interface collecting age, sex, weight (optional), and medication list
- **Model Priority Score**: The probability value (0–100%) from the trained random forest, frozen at 50% threshold
- **No Priority Signal**: Label applied when Model Priority Score is below threshold
- **Review Priority Signal**: Label applied when Model Priority Score is at or above threshold
- **Medication Mapping Coverage**: Table showing matched, ambiguous, and unmatched medication names and their inclusion in model features
- **Frozen Threshold**: The operating decision boundary (50%), loaded from the model bundle or set to 0.50 default
- **Frozen Bundle**: `data/processed/models/imrad_models.joblib`, containing imputer, scaler, random forest, feature_cols, and threshold
- **Drug Dictionary**: `data/processed/drug_dictionary.parquet`, containing faers_raw, dc_id, atc_code, ror, has_boxed_warning columns
- **RapidFuzz Matching**: Fuzzy medication name matching against dictionary using WRatio scorer
- **Feature Defaults**: Train-fitted imputer statistics used to initialize the feature row before user input
- **Loading State**: Indicator shown while model is evaluating or artifacts are being verified
- **Error State**: Message and recovery panel shown when artifacts are missing or validation fails
- **Empty State**: Panel shown when no regimen has been evaluated, containing example buttons and guidance
- **Success State**: Result display including Model Priority Score, threshold, top contributors, and structured guidance
- **Disclaimer**: Required research decision-support statement displayed beside every result
- **Reduced-Motion**: CSS media query (`prefers-reduced-motion`) applied to disable animations for users who opt out
- **Fintech Orange**: Accent color for secondary elements; transitions to clinical green (#2FBF71) on deep green-tinted dark background
- **Clinical Green**: Primary action and active states (#2FBF71)
- **Ink**: Primary text color (#16221c) for accessibility on light backgrounds

## Requirements

### Requirement 1: Home Screen Carousel

**User Story:** As a new visitor, I want to see an engaging carousel of key information so that I understand what TekaRx is, its training data source, what results I'll receive, and why I should use it before entering the predict workflow.

#### Acceptance Criteria

1. WHEN the App loads, THE App SHALL display a carousel on the Home Screen with 3–4 full-width slides.
2. WHEN the carousel is visible, THE App SHALL display left and right arrow buttons positioned at the slide edges, center-aligned vertically.
3. WHEN the visitor clicks a left or right arrow, THE App SHALL advance or retreat one slide with a smooth fade or slide animation.
4. WHILE the carousel is displayed, THE App SHALL display dot indicators below the slides, one dot per slide.
5. WHEN the visitor clicks a dot indicator, THE App SHALL navigate to the corresponding slide.
6. WHEN the visitor hovers over the carousel, THE App SHALL pause animation playback and hold the current slide.
7. WHEN the carousel is paused, THE App SHALL display a visual indicator that playback is paused.
8. WHEN the user's system preference is `prefers-reduced-motion: reduce`, THE App SHALL disable slide animations and carousel autoplay; arrows and dots remain functional.
9. WHEN the carousel has completed its last slide and autoplay is active, THE App SHALL loop back to the first slide.
10. THE App SHALL render all carousel text content free of marketing slop per `antislop.md`: no em dashes, no fabricated claims, no generic buzzwords.

### Requirement 2: Carousel Slide Content

**User Story:** As a new visitor, I want each carousel slide to tell me what TekaRx does, what it's trained on, what results I get, and why I should trust the disclaimer so that I understand the research-only scope before entering my data.

#### Acceptance Criteria

1. WHEN the first carousel slide displays, THE App SHALL show a headline explaining what TekaRx is and a one-line description of the primary use case.
2. WHEN the second carousel slide displays, THE App SHALL show that the model was trained on the FDA FAERS database, the number of reported adverse events, and a note that predictions are research decision-support only.
3. WHEN the third carousel slide displays, THE App SHALL show what the visitor receives: Model Priority Score, medication mapping coverage, and top predictive contributors.
4. WHEN a fourth carousel slide is present, THE App SHALL display the disclaimer, terms of use, and a call-to-action to enter the Predict screen.
5. WHEN any carousel slide displays, THE App SHALL include the one-line research disclaimer: "Research decision-support, not medical advice."
6. WHEN the carousel content is generated, THE App SHALL pass antislop content checks: active voice, no vague terms, specific terminology from the Glossary, no fabricated statistics.

### Requirement 3: Persistent Navigation Bar

**User Story:** As a visitor navigating between Home, Predict, How It Works, and Safety screens, I want a persistent navigation bar that shows which screen I'm on and lets me jump to any other screen with one click.

#### Acceptance Criteria

1. WHEN the App loads on any screen, THE App SHALL render a fixed navigation bar at the top.
2. WHILE the user is on the Home Screen, THE App SHALL display "Home" with an active visual state (bold font, underline, or highlight).
3. WHILE the user is on the Predict Screen, THE App SHALL display "Predict" with an active visual state.
4. WHILE the user is on the How It Works Screen, THE App SHALL display "How It Works" with an active visual state.
5. WHILE the user is on the Safety Screen, THE App SHALL display "Safety" with an active visual state.
6. WHEN the visitor clicks a navigation item, THE App SHALL navigate to that screen and update the active state.
7. WHEN the navigation bar is rendered, THE App SHALL use clinical green (#2FBF71) for active states and a secondary color for inactive items.
8. WHEN the user's keyboard focus enters the navigation bar, THE App SHALL show a visible focus indicator on the active item and allow Tab navigation through all navigation links.
9. WHEN the user presses Tab from a navigation link, THE App SHALL cycle through all navigation items logically.
10. WHEN the viewport is narrower than 768px, THE App SHALL show a hamburger menu or collapsible navigation instead of full-width nav items.

### Requirement 4: Predict Screen Form

**User Story:** As a clinician, I want to enter age, sex, medications, and optionally weight so that the model can evaluate the regimen and return a priority score and clinical insights.

#### Acceptance Criteria

1. WHEN the visitor is on the Predict Screen, THE App SHALL display a form labeled "Patient Regimen Entry".
2. WHEN the form is displayed, THE App SHALL include an input field for age (0–120 years, integer).
3. WHEN the form is displayed, THE App SHALL include a dropdown for biological sex with options: Female, Male, Unknown.
4. WHEN the form is displayed, THE App SHALL include a text area for medication names, accepting one per line or comma-separated.
5. WHERE a weight field is optional, THE App SHALL include a weight input field (in kg, floating-point) that does not block form submission if empty.
6. WHEN all required fields (age, sex, medications) are populated, THE App SHALL enable a "Run TekaRx Score" button.
7. WHEN the "Run TekaRx Score" button is clicked, THE App SHALL validate form inputs and trigger the prediction workflow.
8. WHEN the form is populated with invalid input (e.g., age > 120 or < 0), THE App SHALL show a validation error and prevent submission.
9. WHEN the user focuses on a form field, THE App SHALL display a focus indicator meeting WCAG AA contrast requirements.
10. WHEN the user's system applies reduced motion preference, THE App SHALL not animate form focus or input transitions.

### Requirement 5: Medication Matching Feedback

**User Story:** As a clinician, I want to see which medications were matched to the dictionary, which were ambiguous, and which couldn't be found so that I can confirm the model has coverage over the regimen I entered.

#### Acceptance Criteria

1. WHEN the model processes medications, THE App SHALL fuzzy-match each entered medication name against `drug_dictionary.parquet["faers_raw"]` using RapidFuzz WRatio.
2. WHEN a match score is >= 90%, THE App SHALL classify the match as "Matched" and include the medication in model features.
3. WHEN a match score is >= 75% but < 90%, THE App SHALL classify the match as "Ambiguous" and include the medication in model features.
4. WHEN a match score is < 75%, THE App SHALL classify the match as "Unmatched" and exclude the medication from model features.
5. WHEN the matching is complete, THE App SHALL display a "Medication Mapping Coverage" table with columns: Entered Medication, Mapping Status, Matched Dictionary Name, Match Confidence, Suggested RapidFuzz Correction, Included in Model Features.
6. WHEN any medications are unmatched, THE App SHALL display a warning banner with the count and request the clinician verify spellings using the suggestions provided.
7. WHEN all medications are matched, THE App SHALL display no warning.
8. WHEN the user applies reduced motion preference, THE App SHALL render the mapping table without animation.

### Requirement 6: Model Prediction and Loading State

**User Story:** As a clinician, I want to see a loading indicator while the model is evaluating my regimen so that I know the system is processing and I should wait.

#### Acceptance Criteria

1. WHEN the "Run TekaRx Score" button is clicked, THE App SHALL transition to a loading state.
2. WHILE in the loading state, THE App SHALL display a loading spinner or animated indicator.
3. WHILE in the loading state, THE App SHALL disable the form submit button.
4. WHEN the model completes evaluation within 5 seconds, THE App SHALL clear the loading state and display the success state.
5. IF the model evaluation takes longer than 5 seconds without completing, THE App SHALL show a timeout message and offer a retry button.
6. WHEN the user applies reduced motion preference, THE App SHALL display a static loading indicator with text instead of animation.

### Requirement 7: Model Priority Score Display

**User Story:** As a clinician, I want to see the Model Priority Score as a percentage with a label (Review Priority Signal or No Priority Signal) and the operating threshold so that I understand whether the regimen requires clinical review.

#### Acceptance Criteria

1. WHEN prediction is complete and successful, THE App SHALL display the Model Priority Score as a percentage (0–100%) in large, prominent text.
2. WHEN the Model Priority Score is >= 50% (the frozen threshold), THE App SHALL display the label "Review Priority Signal" in an amber-colored badge.
3. WHEN the Model Priority Score is < 50%, THE App SHALL display the label "No Priority Signal" in a green-colored badge.
4. WHEN the result is displayed, THE App SHALL show a caption: "Operating decision threshold: 50% | Trained Random Forest paradigm".
5. WHEN the result is displayed, THE App SHALL include the one-line disclaimer: "Research decision-support, not medical advice."
6. WHEN the Model Priority Score is displayed, THE App SHALL apply clinical green (#2FBF71) for "No Priority Signal" and fintech orange for "Review Priority Signal".
7. WHEN the result is displayed on a screen narrower than 768px, THE App SHALL stack the score and badge vertically while maintaining readability.

### Requirement 8: Result Guidance and Clinical Context

**User Story:** As a clinician, I want to see the meaning of the score, recommended next steps, and a summary of key medication features so that I can take informed clinical action.

#### Acceptance Criteria

1. WHEN the Model Priority Score is >= 50%, THE App SHALL display guidance: "The model priority score is at or above the frozen operating threshold (50%). Based on the patient demographics and pharmacological features of the reported regimen, this combination exhibits a priority signal consistent with serious adverse event patterns."
2. WHEN the Model Priority Score is >= 50%, THE App SHALL recommend: "Conduct a clinical review of the medication list, paying particular attention to drug interactions, cumulative organ toxicities, and the top contributing factors surfaced below."
3. WHEN the Model Priority Score is < 50%, THE App SHALL display guidance: "The model priority score is below the frozen operating threshold (50%). The combination presents No Priority Signal under the trained IMRAD seriousness model."
4. WHEN the Model Priority Score is < 50%, THE App SHALL recommend: "Document the current regimen and continue standard monitoring. Re-evaluate if medications are added, doses adjusted, or if new symptoms arise."
5. WHEN unmatched medications exist, THE App SHALL append a note to the guidance: "[N] medication(s) were not matched in the clinical dictionary. Verify spellings using the suggestions below to ensure complete model coverage."
6. WHEN the guidance is displayed, THE App SHALL include the disclaimer beside the result.
7. WHEN the user applies reduced motion preference, THE App SHALL render guidance text without animation.

### Requirement 9: Top Contributors and Feature Contribution

**User Story:** As a clinician, I want to see the top five features that contributed most to the Model Priority Score so that I understand which aspects of the regimen drove the signal.

#### Acceptance Criteria

1. WHEN the Model Priority Score is displayed, THE App SHALL render a "Contribution to Model Score" section.
2. WHEN this section is displayed, THE App SHALL show the top 5 features by feature importance from the frozen random forest model.
3. WHEN the top 5 features are displayed, THE App SHALL show three columns: Feature, Regimen Value, Relative Contribution.
4. WHEN a feature is displayed, THE Feature column SHALL show a human-readable label (e.g., "Highest Drug ROR", "Matched Medication Count").
5. WHEN a feature is displayed, THE Regimen Value column SHALL show the patient's specific value for that feature (e.g., "3 medications", "2.45 ROR").
6. WHEN a feature is displayed, THE Relative Contribution column SHALL show the feature's contribution as a percentage of the top 5 combined importance.
7. WHEN this section is displayed, THE App SHALL include a caption: "Top feature contributions from the frozen model pipeline for this specific patient input."
8. WHEN contributions are displayed, THE App SHALL NOT use terms like "SHAP value" or "feature attribution"; only "Contribution to Model Score" is shown.

### Requirement 10: Summary Metrics and Medication Statistics

**User Story:** As a clinician, I want to see at a glance how many medications were matched, what the highest ROR is, and whether there are boxed warnings so that I can quickly assess the regimen complexity.

#### Acceptance Criteria

1. WHEN the result is displayed, THE App SHALL show a summary grid with key metrics.
2. WHEN the summary grid is displayed, THE App SHALL include a "Matched Medications" card showing the count of matched medications.
3. WHEN the summary grid is displayed, THE App SHALL include a "Highest Drug ROR" card showing the maximum relative odds ratio from matched medications.
4. WHEN the summary grid is displayed, THE App SHALL include a "Boxed Warning" card showing "Present" or "None".
5. WHEN the summary grid is displayed on a screen narrower than 768px, THE App SHALL stack cards vertically.
6. WHEN metrics are displayed, THE App SHALL use clear, readable labels and numeric formats (no abbreviations without definition).

### Requirement 11: Error State and Missing Artifacts

**User Story:** As a visitor or system operator, I want a clear message and recovery instructions when the app is missing required artifacts (model bundle or drug dictionary) so that I know how to fix the problem.

#### Acceptance Criteria

1. IF the model bundle (`data/processed/models/imrad_models.joblib`) or drug dictionary (`data/processed/drug_dictionary.parquet`) is missing, THEN THE App SHALL detect this at load time.
2. WHEN artifacts are missing, THE App SHALL display an in-page alert panel with title "Model artifacts are not ready yet".
3. WHEN the error panel is displayed, THE App SHALL list the missing files.
4. WHEN the error panel is displayed, THE App SHALL show exact repository commands to rebuild the artifacts:
   ```
   python -m pip install -e .[imrad,notebook]
   python build_imrad_artifacts.py
   ```
5. WHEN the error panel is displayed, THE App SHALL include the disclaimer: "Research decision-support, not medical advice."
6. WHEN the error panel is displayed, THE App SHALL NOT allow the user to proceed to prediction.

### Requirement 12: Empty State with Example Buttons

**User Story:** As a new clinician, I want to see example regimens so that I can quickly test the app without having to manually enter realistic medication combinations.

#### Acceptance Criteria

1. WHEN the Predict Screen loads and no regimen has been submitted, THE App SHALL display an empty-state panel with title "No Regimen Evaluated Yet".
2. WHEN the empty state is displayed, THE App SHALL include guidance: "Enter patient details and medications in the form on the left, or load one of the clinical test regimens below to inspect the model output."
3. WHEN the empty state is displayed, THE App SHALL show two example buttons: "Load High-Priority Polypharmacy Example" and "Load Maintenance Monotherapy Example".
4. WHEN "Load High-Priority Polypharmacy Example" is clicked, THE App SHALL prefill the form with a realistic high-priority case (e.g., age 74, Female, medications: Warfarin, Ibuprofen, Tramadol, Metoprolol, Omeprazole).
5. WHEN "Load Maintenance Monotherapy Example" is clicked, THE App SHALL prefill the form with a realistic low-priority case (e.g., age 52, Male, medications: Metformin, Atorvastatin, Levothyroxine).
6. WHEN an example button is clicked, THE App SHALL auto-submit the form and display the prediction result.
7. WHEN an example is loaded, THE App SHALL clear session state on subsequent form resets so previous examples do not interfere.

### Requirement 13: All States Exercise with Real Artifacts

**User Story:** As a developer, I want the app to be tested in all UI states (empty, loading, error, missing artifact, success) with real trained model artifacts so that I can trust the implementation is production-ready.

#### Acceptance Criteria

1. WHEN the app is started with the frozen model bundle and drug dictionary present, THE App SHALL load artifacts without error.
2. WHEN artifacts are present and the app completes a prediction, THE App SHALL display the success state with all fields populated from real model output.
3. WHEN the developer removes the model bundle and restarts the app, THE App SHALL detect the missing artifact and display the error panel.
4. WHEN the error panel is displayed, THE App SHALL show the exact file path and recovery commands.
5. WHEN the developer adds the model bundle back and reloads, THE App SHALL recover and display the form again.
6. WHEN the app is on the Predict Screen with no form submission, THE App SHALL display the empty state.
7. WHEN the developer clicks an example button, THE App SHALL display the loading state, then the success state.
8. WHEN all states transition correctly and display real results, THE Delivery Gate in antislop (R-35) has been satisfied.

### Requirement 14: Slop-Free Copy per Antislop

**User Story:** As a product owner, I want the app copy to be free of marketing slop, generic buzzwords, and fabricated claims so that it feels crafted and trustworthy.

#### Acceptance Criteria

1. WHEN any text is generated for the app (buttons, labels, headings, guidance, disclaimers), THE text SHALL NOT contain em dashes (`—`).
2. WHEN any text is generated, THE text SHALL use active voice and specific terminology from the Glossary.
3. WHEN any text is generated, THE text SHALL NOT use vague terms like "quickly", "adequate", "reasonable", "user-friendly", "powerful", "revolutionary", "seamless", or "cutting-edge".
4. WHEN any text is generated, THE text SHALL NOT include fabricated statistics, fake testimonials, or unsupported claims.
5. WHEN any text is generated, THE text SHALL NOT use pronouns; use specific names (e.g., "THE App" not "It", "THE Clinician" not "you").
6. WHEN any copy is drafted, THE copy SHALL pass a manual antislop review before delivery.

### Requirement 15: Accessibility and Keyboard Navigation

**User Story:** As a clinician using keyboard-only input or a screen reader, I want the app to be fully navigable and understandable without a mouse so that I can use it regardless of my input method.

#### Acceptance Criteria

1. WHEN the user presses Tab, THE App SHALL navigate through all interactive elements (buttons, links, form inputs, dropdowns) in logical reading order.
2. WHEN the user focuses on an interactive element, THE App SHALL display a visible focus indicator with contrast ratio >= 4.5:1 (WCAG AA).
3. WHEN the user presses Enter on a button or link, THE element SHALL activate its associated action.
4. WHEN the user presses Space on a checkbox or radio button, THE element SHALL toggle its state.
5. WHEN the carousel is active, THE user SHALL navigate slides using left/right arrow keys in addition to button clicks.
6. WHEN the carousel is active, THE user SHALL navigate to specific slides using number keys (1–4 for slide 1–4).
7. WHEN a form field has focus, THE label SHALL be associated via the `<label for="...">` attribute or semantic HTML.
8. WHEN error messages appear, THE error text SHALL be associated with the field via `aria-describedby`.
9. WHEN the app displays a loading state, THE loading indicator SHALL have `role="status"` and `aria-live="polite"` so screen readers announce the state.
10. WHEN the app displays a result, THE result section SHALL have `role="region"` with an `aria-label` so screen readers identify it as a results area.

### Requirement 16: Reduced-Motion Support

**User Story:** As a clinician with vestibular or motion sensitivity, I want animations to be disabled when I set my system preference to reduce motion so that the app does not trigger discomfort.

#### Acceptance Criteria

1. WHEN the user's system has `prefers-reduced-motion: reduce` set, THE App SHALL detect this preference at load time.
2. WHEN this preference is detected, THE carousel SHALL not autoplay.
3. WHEN this preference is detected, THE carousel slides SHALL not animate; they SHALL change instantly.
4. WHEN this preference is detected, THE loading spinner SHALL be replaced with a static indicator and text.
5. WHEN this preference is detected, THE app SHALL not apply any other CSS animations or transitions.
6. WHEN the user does NOT have `prefers-reduced-motion: reduce` set, THE app MAY apply animations per design.
7. WHEN animations are applied, THE duration SHALL be short (< 300ms) to avoid jarring transitions.

### Requirement 17: Color Scheme and Design Consistency

**User Story:** As a user, I want the app to feel designed, not generic, with a clear identity and color palette so that I trust the clinical scope and professionalism.

#### Acceptance Criteria

1. WHEN the app is rendered, THE primary background color SHALL be white or near-white (#ffffff or #f8faf9).
2. WHEN the app is rendered, THE primary text color SHALL be deep ink (#16221c) for accessibility on light backgrounds.
3. WHEN active or success states are rendered, THE color SHALL be clinical green (#2FBF71).
4. WHEN secondary actions or warnings are rendered, THE color SHALL be fintech orange (specified in design or approximated as #8a580a).
5. WHEN the app is in a light theme, THE design SHALL maintain WCAG AA contrast for all text on backgrounds.
6. WHEN form elements are rendered, THE border color SHALL be a neutral grey (#d8e2dc) that contrasts with the background.
7. WHEN the app is rendered, THE design SHALL NOT use multiple unrelated color schemes or trend-stacked effects (glassmorphism + mesh gradient + glow all at once).
8. WHEN the design is applied, ALL interactive elements SHALL have consistent styling (buttons, links, dropdowns, inputs, checkboxes).

### Requirement 18: README Documentation

**User Story:** As a developer or operator, I want clear run-and-setup instructions in the README so that I can get the app running locally and know what commands to use.

#### Acceptance Criteria

1. WHEN the README is read, THE run-steps section SHALL include:
   ```
   python -m pip install -r frontend/requirements.txt
   streamlit run frontend/streamlit_app.py
   ```
2. WHEN the README is read, THE setup section SHALL list the required artifacts and where they come from.
3. WHEN the README is read, THE recovery section SHALL show the build commands if artifacts are missing.
4. WHEN the README is read, THE section SHALL mention the antislop review and that copy has been checked for marketing slop.
5. WHEN the README is read, THE documentation SHALL include a link to `docs/streamlit_dashboard.md` for implementation details.

### Requirement 19: Code Quality and Linting

**User Story:** As a developer, I want the code to pass all linting checks so that the codebase is clean and consistent.

#### Acceptance Criteria

1. WHEN the code is run through `ruff check`, THE linter SHALL report no errors or warnings.
2. WHEN the code is run through `ruff format`, THE formatter SHALL produce no changes (code is already formatted).
3. WHEN the code is committed, THE code SHALL follow the existing `frontend/streamlit_app.py` style and patterns.

### Requirement 20: Preserved Accuracy and Model Integrity

**User Story:** As a clinician, I want the app to load the exact frozen model artifacts and run predictions without retraining or fabricating inputs so that results are trustworthy and reproducible.

#### Acceptance Criteria

1. WHEN the app loads, THE model bundle SHALL be loaded from `data/processed/models/imrad_models.joblib` without modification.
2. WHEN the app loads, THE feature_cols order from the bundle SHALL be used exactly; no reordering or transformation.
3. WHEN the app builds features from user input, THE imputer statistics (train-fitted) SHALL initialize missing columns.
4. WHEN the app transforms features, THE scaler from the bundle SHALL be applied to the imputed frame.
5. WHEN the app predicts, THE random_forest model SHALL run predict_proba without retraining.
6. WHEN the app matches medications, THE RapidFuzz matching SHALL use the WRatio scorer against `drug_dictionary.parquet["faers_raw"]`.
7. WHEN the app displays results, THE threshold SHALL be loaded from the bundle or set to 0.50 if no threshold key is found.
8. WHEN the app builds the feature row, THE ROR, ATC, and boxed-warning fields SHALL be extracted from matched dictionary rows only.
9. WHEN the app runs a prediction, THE result SHALL match the Python reproduction example in `docs/streamlit_dashboard.md` exactly.
