# Implementation Plan

- [ ] 1. Write bug condition exploration test
  - **Property 1: Bug Condition** - Timeline Endpoint 404
  - **CRITICAL**: This test MUST FAIL on unfixed code - failure confirms the bug exists
  - **DO NOT attempt to fix the test or the code when it fails**
  - **NOTE**: This test encodes the expected behavior - it will validate the fix when it passes after implementation
  - **GOAL**: Surface counterexamples that demonstrate the bug exists
  - Test that GET /timeline returns 200 OK with timeline data (not 404)
  - Run test on UNFIXED code
  - **EXPECTED OUTCOME**: Test FAILS with 404 (this confirms the bug exists)
  - Document counterexamples found (e.g., "GET /timeline returns 404 Not Found")
  - _Requirements: 1.1, 1.2_

- [ ] 2. Write preservation property tests (BEFORE implementing fix)
  - **Property 2: Preservation** - Other Endpoints Continue Working
  - **IMPORTANT**: Follow observation-first methodology
  - Observe behavior on UNFIXED code for other endpoints
  - Test that GET /health returns 200 OK
  - Test that GET /papers returns 200 OK
  - Test that GET /report returns 200 OK
  - Run tests on UNFIXED code
  - **EXPECTED OUTCOME**: Tests PASS (this confirms baseline behavior to preserve)
  - _Requirements: 3.1, 3.2_

- [ ] 3. Fix for timeline endpoint 404

  - [ ] 3.1 Implement the fix
    - Add import: `from council_api.timeline_router import router as timeline_router`
    - Add router registration: `app.include_router(timeline_router)`
    - _Bug_Condition: timeline_router not registered in main.py_
    - _Expected_Behavior: expectedBehavior(result) - /timeline returns 200 with timeline data_
    - _Preservation: Other endpoints continue to work as before_
    - _Requirements: 1.1, 1.2, 2.1, 2.2, 3.1, 3.2_

  - [ ] 3.2 Verify bug condition exploration test now passes
    - **Property 1: Expected Behavior** - Timeline Endpoint Works
    - **IMPORTANT**: Re-run the SAME test from task 1 - do NOT write a new test
    - Run bug condition exploration test from step 1
    - **EXPECTED OUTCOME**: Test PASSES (confirms bug is fixed)
    - _Requirements: 2.1, 2.2_

  - [ ] 3.3 Verify preservation tests still pass
    - **Property 2: Preservation** - Other Endpoints Still Work
    - **IMPORTANT**: Re-run the SAME tests from task 2 - do NOT write new tests
    - Run preservation property tests from step 2
    - **EXPECTED OUTCOME**: Tests PASS (confirms no regressions)
    - Confirm all tests still pass after fix (no regressions)
    - _Requirements: 3.1, 3.2_

- [ ] 4. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.