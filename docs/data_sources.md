# Data sources and provenance

Raw downloads are immutable inputs. Each extractor records its URL, retrieval timestamp,
byte size, and SHA-256 checksum in `data/raw/<source>/manifest.json`. Never commit raw or
processed healthcare data to this repository.

## FAERS / AEMS

- Publisher: U.S. Food and Drug Administration (FDA)
- Source: https://www.fda.gov/drugs/fda-adverse-event-monitoring-system-aems/fda-adverse-event-monitoring-system-aems-latest-quarterly-data-files
- Format retained: official quarterly ASCII ZIP and extracted source files
- Important limitation: spontaneous reports can be duplicated, incomplete, biased, or
  unverified. They do not establish that a drug caused an event.
- Terms: review the FDA website policies and the README distributed with every quarter.

## DrugCentral

- Publisher: University of New Mexico, Division of Translational Informatics
- Source: https://drugcentral.org/download
- Format retained: official PostgreSQL dump
- License: Creative Commons Attribution-ShareAlike 4.0 according to DrugCentral's published
  data-availability documentation. Preserve attribution and verify the current license at
  download time.

## DailyMed

- Publisher: U.S. National Library of Medicine
- Source: https://dailymed.nlm.nih.gov/dailymed/spl-resources.cfm
- Formats retained: official mapping ZIP files and, when needed, original SPL XML ZIP files
- Terms: review NLM copyright and usage policies and retain source attribution.

## RxNorm / RxNav API

- Publisher: U.S. National Library of Medicine
- API documentation: https://lhncbc.nlm.nih.gov/RxNav/APIs/RxNormAPIs.html
- Terms: https://lhncbc.nlm.nih.gov/RxNav/TermsofService.html
- Access: free of charge and no key is required for the endpoints used here. NLM limits clients
  to 20 requests per second per IP; TekaRx defaults to 10 and caches batch responses.
- Provenance: the lookup manifest records the DrugCentral dump checksum, build time, API endpoint,
  and whether the API fallback was enabled. RxNorm changes over time, so retain the manifest with
  experiment artifacts.

## Modeling boundary

DailyMed label versions published after a FAERS report date must not be used as historical
predictors for that report. Outcome and reaction fields are labels or post-event information,
not baseline predictors.

FAERS serious-outcome categories are `DE`, `LT`, `HO`, `DS`, `CA`, `RI`, and `OT`. TekaRx
defines `is_serious = 1` when any of these categories is documented and zero when no serious
outcome is documented. The label means *reported seriousness*; it is not proof that a drug
caused an event and it is not an estimate of population incidence.

`has_boxed_warning` in the static drug dictionary is derived from the DrugCentral dump's
DailyMed-ingested SPL sections using LOINC `34066-1`. Because this is not a report-date-versioned
feature, it must not be used as a historical predictor without confirming that the warning was
effective on or before each FAERS report date.

### Dosage interpretation

FAERS dosage fields are reporter-entered exposure descriptions. They can be incomplete,
inconsistent, or product-level rather than ingredient-level, and must not be interpreted as
verified administered doses.

TekaRx only converts compatible physical dimensions. Mass, mass per body weight, mass per body
surface area, volume, international units, milliequivalents, and amount of substance remain
separate. In particular, the pipeline does not convert volume to mass without a concentration,
does not convert international units to mass, and does not infer an amount from a percentage,
tablet count, spray, or other presentation unit. A daily amount is derived only when both the
amount and an unambiguous administration frequency are available.

Drug-relative normalization is fitted on training reports only and then frozen for validation
and test. It is available only when a FAERS product name maps unambiguously to one DrugCentral
ingredient. A combination product's total reported dose is never copied to each constituent
ingredient; such rows remain unavailable for ingredient-relative dose features unless an
ingredient-specific source amount is present.
