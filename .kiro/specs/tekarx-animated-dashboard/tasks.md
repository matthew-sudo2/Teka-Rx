# Implementation Plan: TekaRx Animated Dashboard

## Overview

Convert the TekaRx Streamlit dashboard from a single-screen form into a multi-screen application with a carousel-based home screen, persistent navigation, organized state management, and enhanced medication matching feedback. The implementation preserves all accuracy contracts (frozen models, RapidFuzz matching, feature column order) while adding motion, accessibility, and slop-free copy. All implementation uses Python (Streamlit) and follows existing code patterns in `frontend/streamlit_app.py`.

## Tasks

- [ ] 1. Set up session state, navigation routing, and carousel state management
  - Create session state entries for `current_page`, `carousel_index`, `carousel_paused`, `form_data`, `prediction_result`
  - Implement URL hash-based routing to enable browser back/forward
  - Implement `_get_active_page()` function to return current page (Home, Predict, How It Works, Safety)
  - Set up session initialization on app load to reset state only when necessary
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_

- [ ] 2. Implement persistent navigation bar with active state tracking
  - [ ] 2.1 Create `_render_navigation()` function
    - Render fixed top navigation bar with 4 items: Home, Predict, How It Works, Safety
    - Apply active visual state (clinical green #2FBF71, bold) to current page
    - Add click handlers to navigate between pages
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 3.10_
  
  - [ ]* 2.2 Write property test for active navigation state
    - **Property 2: Active Navigation State Matches Screen Context**
    - **Validates: Requirements 3.2, 3.3, 3.4, 3.5, 3.6_

- [ ] 3. Implement carousel component with navigation and autoplay
  - [ ] 3.1 Create carousel data structure and slide content
    - Define slide 1: "What is TekaRx?" with headline and one-liner about medication safety
    - Define slide 2: "Trained on FAERS" with dataset info and research-only disclaimer
    - Define slide 3: "What You Get" with Model Priority Score, medication mapping, top contributors
    - Define slide 4: "Disclaimer & Next Steps" with terms, research use, and CTA
    - All text must pass antislop review (no em dashes, no vague terms, specific terminology)
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 14.1, 14.2, 14.3, 14.4, 14.5_
  
  - [ ] 3.2 Create `_render_carousel()` function
    - Display 4 full-width slides with centered text
    - Render left/right arrow buttons at slide edges, vertically centered
    - Render dot indicators below slides (one per slide), clickable to navigate
    - Implement fade transition animation (300ms default)
    - Implement carousel autoplay with 4-second interval per slide
    - Implement pause-on-hover with visual indicator
    - Loop back to first slide after last slide when autoplay is active
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.9_
  
  - [ ] 3.3 Implement reduced-motion support for carousel
    - Detect `prefers-reduced-motion: reduce` at app load
    - Disable autoplay when reduced motion is detected
    - Disable slide animations (instant slide change) when reduced motion is detected
    - Keep arrows and dots functional
    - _Requirements: 1.8, 16.2, 16.3, 16.4, 16.5_
  
  - [ ] 3.4 Implement carousel keyboard navigation
    - Left/Right arrow keys: Navigate slides
    - Home key: Jump to slide 1
    - End key: Jump to last slide
    - Number keys (1-4): Jump to specific slide
    - _Requirements: 15.5, 15.6_
  
  - [ ]* 3.5 Write property tests for carousel navigation
    - **Property 1: Carousel Navigation is Reversible**
    - **Validates: Requirements 1.3, 1.4, 1.5, 1.9**
    - **Property 7: Reduced Motion Disables Carousel Autoplay**
    - **Validates: Requirements 1.8, 16.2, 16.3**
    - **Property 18: Carousel Slides Display Correct Content**
    - **Validates: Requirements 2.1, 2.2, 2.3, 2.4**

- [ ] 4. Implement Home Screen with carousel and static navigation
  - [ ] 4.1 Create `_render_home_screen()` function
    - Render carousel component from task 3
    - Render static sections: introduction, research mission statement
    - Include CTA button "Enter Predict Screen" below carousel
    - Apply clinical green and fintech orange color scheme
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 17.1, 17.3, 17.4_
  
  - [ ]* 4.2 Write unit tests for Home Screen rendering
    - Test carousel renders all 4 slides
    - Test CTA button navigates to Predict Screen
    - Test color scheme is applied
    - _Requirements: 2.1, 2.2, 2.3, 2.4_

- [ ] 5. Create How It Works and Safety screens
  - [ ] 5.1 Create `_render_how_it_works_screen()` function
    - Display static educational content explaining the model, terminology, and FAERS dataset
    - Include sections: What is a Model Priority Score, What is Medication Mapping, How are Features Computed
    - Use active voice, specific terminology from Glossary
    - _Requirements: 2.3, 14.1, 14.2, 14.3_
  
  - [ ] 5.2 Create `_render_safety_screen()` function
    - Display disclaimer, terms of use, and data handling practices
    - Include research decision-support statement
    - Display artifact load status for operator debugging
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 11.6_

- [ ] 6. Implement Predict Screen form layout with left/right columns
  - [ ] 6.1 Create `_render_predict_screen()` function
    - Left column (50%): Form inputs for age, sex, weight, medications
    - Right column (50%): Empty state, loading state, or results state
    - Use responsive layout (stack vertically on mobile < 768px)
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.7_
  
  - [ ] 6.2 Create form input fields with validation
    - Age field: Integer input 0-120 with validation
    - Sex field: Dropdown (Female, Male, Unknown)
    - Weight field: Optional float input (0-500 kg)
    - Medications field: Text area (one per line or comma-separated)
    - All form inputs use consistent styling and focus indicators
    - _Requirements: 4.2, 4.3, 4.4, 4.5, 4.6, 4.8, 4.9, 4.10_
  
  - [ ]* 6.3 Write property test for form validation
    - **Property 8: Form Submission is Blocked on Invalid Input**
    - **Validates: Requirements 4.6, 4.8**

- [ ] 7. Implement empty state with example buttons
  - [ ] 7.1 Create `_render_empty_state()` function
    - Display title "No Regimen Evaluated Yet"
    - Display guidance text directing user to enter data or load example
    - Render two example buttons: "Load High-Priority Polypharmacy Example", "Load Maintenance Monotherapy Example"
    - Style with neutral border and light background
    - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5_
  
  - [ ] 7.2 Create example data and autoload logic
    - High-priority example: age 74, Female, medications [Warfarin, Ibuprofen, Tramadol, Metoprolol, Omeprazole]
    - Low-priority example: age 52, Male, medications [Metformin, Atorvastatin, Levothyroxine]
    - On click, prefill form and auto-submit
    - Clear session state after example loads to prevent interference
    - _Requirements: 12.4, 12.5, 12.6, 12.7_
  
  - [ ]* 7.3 Write unit tests for empty state
    - Test empty state renders only before first prediction
    - Test example buttons prefill form and auto-submit
    - _Requirements: 12.1, 12.3, 12.6_

- [ ] 8. Implement loading state with reduced-motion support
  - [ ] 8.1 Create `_render_loading_state()` function
    - Display loading indicator (animated spinner by default)
    - Include message: "Evaluating regimen..."
    - Disable form submit button
    - Implement 5-second timeout with retry button
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_
  
  - [ ] 8.2 Implement reduced-motion loading indicator
    - Replace animated spinner with static icon + text for `prefers-reduced-motion`
    - Message: "Evaluating regimen (model running)..."
    - _Requirements: 6.6, 16.4_

- [ ] 9. Implement medication matching with RapidFuzz feedback
  - [ ] 9.1 Extend existing `_match_medications()` to track classification and confidence
    - Match each medication against `drug_dictionary.parquet["faers_raw"]` using RapidFuzz WRatio
    - Classify as Matched (score >= 90), Ambiguous (75 <= score < 90), Unmatched (score < 75)
    - Track match confidence as percentage
    - Return matched medications for feature building
    - _Requirements: 5.1, 5.2, 5.3, 5.4_
  
  - [ ] 9.2 Create `_render_medication_mapping_table()` function
    - Display table with columns: Entered Medication, Mapping Status, Matched Dictionary Name, Match Confidence, Suggested RapidFuzz Correction, Included in Model Features
    - Show status as "Matched" (green), "Ambiguous" (amber), or "Unmatched" (red badge)
    - Show top 3 RapidFuzz suggestions for unmatched medications
    - _Requirements: 5.5, 5.6, 5.7, 5.8_
  
  - [ ] 9.3 Implement unmatched medication warning banner
    - If any medications unmatched, display warning: "[N] medication(s) could not be mapped with >=90% confidence. Please review suggestions or verify spellings."
    - Warning not displayed if all matched
    - _Requirements: 5.6, 5.7_
  
  - [ ]* 9.4 Write property test for medication matching classification
    - **Property 4: Medication Matching Classification is Consistent**
    - **Validates: Requirements 5.2, 5.3, 5.4**
    - **Property 5: Unmatched Medication Warning Presence is Correct**
    - **Validates: Requirements 5.6, 5.7**
    - **Property 16: RapidFuzz Matching Uses Exact Scorer and Source**
    - **Validates: Requirements 5.1, 20.6**

- [ ] 10. Implement Model Priority Score display with label and threshold
  - [ ] 10.1 Create `_render_priority_score()` function
    - Display probability as large percentage (0-100%)
    - Apply label "Review Priority Signal" (amber badge) if score >= 50%
    - Apply label "No Priority Signal" (green badge) if score < 50%
    - Show caption: "Operating decision threshold: 50% | Trained Random Forest paradigm"
    - Include disclaimer: "Research decision-support, not medical advice."
    - Use clinical green (#2FBF71) for < 50%, fintech orange for >= 50%
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7_
  
  - [ ]* 10.2 Write property test for score label determination
    - **Property 3: Model Priority Score Label is Determined by Threshold**
    - **Validates: Requirements 7.2, 7.3, 7.6**

- [ ] 11. Implement result guidance and clinical context
  - [ ] 11.1 Create `_render_guidance()` function
    - If score >= 50%: Display meaning about priority signal and serious adverse event patterns
    - If score >= 50%: Recommend clinical review focusing on drug interactions and cumulative toxicities
    - If score < 50%: Display meaning about no priority signal
    - If score < 50%: Recommend continued standard monitoring
    - If unmatched medications exist: Append note about incomplete coverage
    - Include disclaimer beside guidance
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7_

- [ ] 12. Implement top contributors table with feature labels
  - [ ] 12.1 Create feature label mapping dictionary
    - Map internal feature names to human-readable labels (e.g., `max_ror` → "Highest Drug ROR")
    - Create `_feature_label()` function to return human-readable label
    - _Requirements: 9.4_
  
  - [ ] 12.2 Create `_render_contributors_table()` function
    - Extract top 5 features by importance from frozen random forest
    - Display table with columns: Feature, Regimen Value, Relative Contribution (%)
    - Compute relative contribution as percentage of top-5 combined importance
    - Show regimen-specific values (e.g., "3 medications", "2.45 ROR")
    - Include caption: "Top feature contributions from the frozen model pipeline for this specific patient input."
    - Use only "Contribution to Model Score" terminology (no SHAP, attribution, or interpretation)
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8_
  
  - [ ]* 12.3 Write property test for feature contributions
    - **Property 6: Feature Importance Contributions Sum Correctly**
    - **Validates: Requirements 9.6**

- [ ] 13. Implement summary metrics grid with key regimen statistics
  - [ ] 13.1 Create `_render_summary_metrics()` function
    - Display 3-card grid: "Matched Medications" (count), "Highest Drug ROR" (value), "Boxed Warning" (Present/None)
    - Stack vertically on mobile (< 768px)
    - Use clear, readable labels and numeric formats
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6_
  
  - [ ]* 13.2 Write unit tests for summary metrics
    - Test exactly 3 cards are displayed
    - Test values are correct for given medication set
    - _Requirements: 10.1, 10.2, 10.3, 10.4_

- [ ] 14. Implement success state with all result components
  - [ ] 14.1 Create `_render_success_state()` function
    - Combine all result components: priority score, guidance, medication mapping, contributors, summary metrics
    - Apply responsive layout (stack on mobile)
    - Include disclaimer beside each major section
    - Display result in right column of Predict Screen
    - _Requirements: 7.1, 8.1, 9.1, 10.1, 12.1_
  
  - [ ]* 14.2 Write integration tests for success state
    - Test all components render correctly with real model predictions
    - Test data flows from form input through prediction to display
    - _Requirements: 13.1, 13.2, 13.3, 13.4, 13.5, 13.6, 13.7, 13.8_

- [ ] 15. Implement error state with artifact missing detection
  - [ ] 15.1 Enhance `_artifact_status()` to check for missing files at app startup
    - Detect missing model bundle or drug dictionary
    - Store status in session state on load
    - _Requirements: 11.1_
  
  - [ ] 15.2 Create `_render_error_state()` function
    - Display title: "Model artifacts are not ready yet"
    - List missing file paths
    - Show exact recovery commands (exact copy from BUILD_COMMANDS)
    - Include disclaimer
    - Block access to prediction until artifacts exist
    - _Requirements: 11.2, 11.3, 11.4, 11.5, 11.6_

- [ ] 16. Implement keyboard navigation and accessibility support
  - [ ] 16.1 Create keyboard event handler for carousel and form navigation
    - Detect arrow key presses for carousel navigation
    - Detect number keys (1-4) for direct slide navigation
    - Detect Home/End keys for carousel edge navigation
    - Detect Tab key for logical focus order
    - _Requirements: 15.1, 15.2, 15.5, 15.6_
  
  - [ ] 16.2 Add focus indicators and ARIA labels to all interactive elements
    - Add 3px solid outline (#b7dbca) focus indicator on buttons, links, inputs, carousel arrows
    - Ensure focus contrast ratio >= 4.5:1 (WCAG AA)
    - Add `aria-current="page"` to active navigation item
    - Add `aria-describedby` to form fields with error messages
    - Add `role="status"` with `aria-live="polite"` to loading indicator
    - Add `role="region"` with `aria-label` to result section
    - _Requirements: 15.1, 15.2, 15.3, 15.4, 15.7, 15.8, 15.9, 15.10_
  
  - [ ]* 16.3 Write property test for accessibility
    - **Property 11: Accessibility Focus Indicators are Always Visible**
    - **Validates: Requirements 15.2, 15.7**
    - **Property 12: Keyboard Tab Navigation is Consistent**
    - **Validates: Requirements 15.1, 15.9**
    - **Property 13: Sticky Navigation Persists Across Screen Transitions**
    - **Validates: Requirements 3.1**

- [ ] 17. Implement color scheme and styling consistency
  - [ ] 17.1 Define color variables and apply throughout app
    - Clinical green (#2FBF71) for active/success states
    - Fintech orange (#8a580a) for warnings/secondary actions
    - Deep ink (#16221c) for primary text
    - Secondary ink (#485750) for secondary text
    - Neutral grey (#d8e2dc) for borders
    - Verify WCAG AA contrast for all text on backgrounds (>= 4.5:1)
    - _Requirements: 17.1, 17.2, 17.3, 17.4, 17.5, 17.6, 17.7_
  
  - [ ] 17.2 Create consistent button and form component styles
    - Button: 6px border-radius, 2.85rem min-height, green bg with white text
    - Form input: 1px border (neutral grey), 6px radius, padding 0.75rem
    - Card: 7px radius, 1px border, light panel background
    - Disclaimer: Left border (3px green), left padding 0.65rem
    - Apply transitions (0.15s) smoothly across all elements
    - _Requirements: 17.1, 17.8_

- [ ] 18. Implement model prediction with frozen artifacts
  - [ ] 18.1 Extend `_build_features()` to handle feature defaults for unmapped medications
    - Load train-fitted imputer statistics from frozen bundle
    - Initialize missing feature columns with imputer statistics
    - Build feature row in exact feature_cols order
    - Extract ROR, ATC, and boxed-warning fields from matched dictionary rows only
    - _Requirements: 20.3, 20.4, 20.8, 20.9_
  
  - [ ] 18.2 Create `_predict()` function
    - Load imputer, scaler, and model from frozen bundle
    - Apply imputer to feature row (train-fitted statistics)
    - Apply scaler to imputed row (frozen statistics)
    - Call model.predict_proba() to get probability
    - Return probability and feature importances
    - _Requirements: 20.1, 20.2, 20.3, 20.4, 20.5_
  
  - [ ]* 18.3 Write property test for model prediction accuracy
    - **Property 9: Model Prediction Outputs Match Artifact Execution**
    - **Validates: Requirements 20.2, 20.3, 20.4, 20.5, 20.9**
    - **Property 14: Feature Defaults Preserve Train-Fitted Imputer Statistics**
    - **Validates: Requirements 20.3**
    - **Property 20: Summary Metrics Grid Contains Exactly Three Cards**
    - **Validates: Requirements 10.1, 10.2, 10.3, 10.4**

- [ ] 19. Refactor main app layout to support multi-screen navigation
  - [ ] 19.1 Restructure `main()` function to route between screens
    - Check artifact status at app start
    - If artifacts missing: Display error state
    - If artifacts present: Display navigation and route to current page
    - Update session state on page navigation
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 11.1, 11.2, 11.3, 11.4, 11.5, 11.6_

- [ ] 20. Implement reduced-motion support across all screens
  - [ ] 20.1 Create `_prefers_reduced_motion()` function
    - Detect `prefers-reduced-motion: reduce` system preference
    - Return boolean flag for use in conditional rendering
    - _Requirements: 16.1_
  
  - [ ] 20.2 Apply reduced-motion CSS throughout app
    - Disable carousel autoplay when reduced-motion is detected
    - Disable slide animations (instant change) when reduced-motion is detected
    - Replace animated loading spinner with static indicator and text
    - Disable all other transitions and animations (transition: none)
    - Keep focus outlines and interactive elements fully functional
    - _Requirements: 16.1, 16.2, 16.3, 16.4, 16.5, 16.6, 16.7_

- [ ] 21. Review copy for slop and pass antislop check
  - [ ] 21.1 Audit all user-facing copy
    - Check all carousel slides: no em dashes, active voice, specific terminology
    - Check all form labels and buttons: no vague terms, specific language
    - Check guidance text: no fabricated claims, research decision-support framing
    - Check disclaimers: consistent research-only messaging
    - Review against antislop.md for marketing slop violations
    - _Requirements: 14.1, 14.2, 14.3, 14.4, 14.5_
  
  - [ ] 21.2 Confirm copy passes team review
    - Copy has been reviewed and approved to be slop-free per antislop.md
    - No em dashes, no vague terms, no fabricated statistics
    - _Requirements: 14.1, 14.2, 14.3, 14.4, 14.5, 13.8_

- [ ] 22. Checkpoint - Ensure all unit and property tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 23. Update README with run steps and artifact documentation
  - [ ] 23.1 Add run-and-setup section
    - Include installation command: `python -m pip install -r frontend/requirements.txt`
    - Include run command: `streamlit run frontend/streamlit_app.py`
    - Include artifact setup explanation
    - _Requirements: 18.1, 18.2_
  
  - [ ] 23.2 Add recovery and debugging section
    - Show build commands if artifacts are missing
    - Link to `docs/streamlit_dashboard.md` for implementation details
    - Mention antislop review and copy validation
    - _Requirements: 18.3, 18.4, 18.5_

- [ ] 24. Run linting and format checks
  - [ ] 24.1 Run `ruff check` on frontend/streamlit_app.py
    - Resolve all errors and warnings
    - Code must follow existing style and patterns
    - _Requirements: 19.1_
  
  - [ ] 24.2 Run `ruff format` on frontend/streamlit_app.py
    - Verify no changes are needed (code is already formatted)
    - _Requirements: 19.2_

- [ ] 25. Final checkpoint - Verify all states with real artifacts
  - Exercise empty state, loading state, success state, and error state with real model artifacts
  - Verify all accuracy contracts preserved (frozen threshold, feature column order, RapidFuzz matching)
  - Verify carousel works on all slides (all states)
  - Verify navigation bar active states work correctly
  - Verify all copy is slop-free
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP (though property tests are critical for accuracy validation)
- Each task references specific requirements for traceability
- All implementation builds on existing `frontend/streamlit_app.py` functions; preserve existing functions and extend them
- Checkpoint tasks validate that core functionality works before proceeding to next phase
- Property tests validate universal correctness properties from the design document
- The app must load real frozen model artifacts—no fabrication or retraining allowed
- All copy must pass antislop review before delivery

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1", "2.1", "3.1"] },
    { "id": 1, "tasks": ["2.2", "3.2", "3.3"] },
    { "id": 2, "tasks": ["3.4", "3.5", "4.1"] },
    { "id": 3, "tasks": ["4.2", "5.1", "5.2"] },
    { "id": 4, "tasks": ["6.1", "6.2", "6.3"] },
    { "id": 5, "tasks": ["7.1", "7.2", "7.3"] },
    { "id": 6, "tasks": ["8.1", "8.2", "9.1"] },
    { "id": 7, "tasks": ["9.2", "9.3", "9.4"] },
    { "id": 8, "tasks": ["10.1", "10.2", "11.1"] },
    { "id": 9, "tasks": ["12.1", "12.2", "12.3"] },
    { "id": 10, "tasks": ["13.1", "13.2", "14.1"] },
    { "id": 11, "tasks": ["14.2", "15.1", "15.2"] },
    { "id": 12, "tasks": ["16.1", "16.2", "16.3"] },
    { "id": 13, "tasks": ["17.1", "17.2", "18.1"] },
    { "id": 14, "tasks": ["18.2", "18.3", "19.1"] },
    { "id": 15, "tasks": ["20.1", "20.2", "21.1"] },
    { "id": 16, "tasks": ["21.2", "23.1"] },
    { "id": 17, "tasks": ["23.2", "24.1"] },
    { "id": 18, "tasks": ["24.2", "25"] }
  ]
}
```
