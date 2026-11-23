# MHCflurry 2.3.9

Download inventory now distinguishes standalone bundles from components inside
presentation bundles. Affinity and processing rows can show `2.3.0 via
presentation`; detailed inspection shows each component's exact path and
whether its manifest is present, including incomplete processing variants.

Existing standalone installation fields retain their meaning in JSON output;
component metadata is additive. Inspection remains offline and does not load
or modify weights. Model selection, environment overrides and prediction
behavior are unchanged. Default weights remain 2.3.0.
