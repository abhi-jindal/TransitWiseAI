---
inclusion: always
---

# TransitWise AI Project Guidance

TransitWise AI is a Streamlit application developed for the
Hackachusetts 2026 Civic Action challenge.

## Project conventions

- Use official MBTA GTFS data for scheduled journeys.
- Support Red, Orange, Blue, and Green subway lines.
- Preserve existing routing and transfer calculations.
- Distinguish scheduled, live, historical, simulated,
  and experimental information.
- Never invent live predictions or statistical confidence.
- Historical data must be specific to the station and line.
- Experimental ML predictions must not be presented as
  validated journey arrival predictions.
- Run existing tests after changing application code.

## Development approach

The project was developed iteratively using Kiro IDE,
including conversational implementation, debugging,
testing, and UI improvements.

This steering document was added during submission
preparation to document current project conventions.
