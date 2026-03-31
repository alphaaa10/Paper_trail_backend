# Bugfix Requirements Document

## Introduction

The `/timeline` endpoint returns 404 Not Found even though the timeline router is implemented in `council_api/timeline_router.py`. This prevents the frontend from accessing the timeline feature which groups papers by year with AI-generated contributions.

## Bug Analysis

### Current Behavior (Defect)

1.1 WHEN a GET request is made to `/timeline` THEN the system returns 404 Not Found
1.2 WHEN a GET request is made to `/timeline` THEN the FastAPI router does not recognize the path because the router is not registered in the main application

### Expected Behavior (Correct)

2.1 WHEN a GET request is made to `/timeline` THEN the system SHALL return timeline data with papers grouped by year
2.2 WHEN a GET request is made to `/timeline` THEN the response SHALL include years, papers_by_year, total_papers, and year_range fields

### Unchanged Behavior (Regression Prevention)

3.1 WHEN requests are made to other endpoints (e.g., `/health`, `/papers`, `/report`) THEN the system SHALL CONTINUE TO work as before
3.2 WHEN requests are made to other registered routers (accuracy, citation, debate, heatmap, qa, browse, crawl_visual) THEN the system SHALL CONTINUE TO work as before