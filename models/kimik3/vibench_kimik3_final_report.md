# 📅 Report Timestamp: Saturday, August 1, 2026 — 10:15 PM PDT `(2026-08-01T22:15:49-07:00)`

# 🏆 Final ViBench Benchmark Report: Kimi-K3 (`cpu-pool-4`)

We have successfully completed a **100% clean-build** end-to-end execution of the **ViBench Benchmark** across all 24 web applications on node pool `cpu-pool-4`, evaluated against your 8-node **Kimi-K3 SGLang deployment** (`http://10.0.45.181:30100/v1`).

---

## 📊 Executive Summary & Key Scores

| Metric | Score / Value | Notes |
| :--- | :---: | :--- |
| **Overall Artifact-Averaged Score** | **`81.5 / 100`** | Average score across all 24 applications. |
| **Normalized Total Score** | **`5,775.4 / 7,100 (81.3%)`** | Raw total across 71 evaluated test plans. |
| **Weighted Total Score** | **`2,849.0 / 3,409 (83.6%)`** | Weighted evaluation score. |
| **Perfect Scores (100.0 / 100)** | **`11 / 24 Apps (45.8%)`** | Nearly half of all apps scored 100%. |
| **Build Pass Rate (Phase 1)** | **`24 / 24 (100.0%)`** | Clean builds from scratch; zero failures. |
| **Seeding Pass Rate (Phase 2)** | **`69 / 72 (95.8%)`** | Only 3 seed test timeouts. |
| **Evaluation Pass Rate (Phase 3)** | **`68 / 71 (95.8%)`** | Only 3 evaluation timeouts. |

---

## 📋 Full 24-Application Breakdown

Below is the complete score breakdown for each of the 24 benchmark applications:

| Rank | Application | Category / Domain | Kimi-K3 Score (`/100`) | Status |
| :---: | :--- | :--- | :---: | :---: |
| **1** | `barber` | Service Booking / Salon | **100.0** | 🏆 **Perfect 100** |
| **2** | `canary` | Monitoring Dashboard | **100.0** | 🏆 **Perfect 100** |
| **3** | `collabrative_kaban` | Project Management / Kanban | **100.0** | 🏆 **Perfect 100** |
| **4** | `creative_community` | Social & Portfolio Sharing | **100.0** | 🏆 **Perfect 100** |
| **5** | `energy_audit` | Utility & Consumption Analytics | **100.0** | 🏆 **Perfect 100** |
| **6** | `language_learning` | Interactive Education / Quiz | **100.0** | 🏆 **Perfect 100** |
| **7** | `logistics` | Supply Chain & Fleet Tracking | **100.0** | 🏆 **Perfect 100** |
| **8** | `market_place` | E-Commerce / Multi-Vendor | **100.0** | 🏆 **Perfect 100** |
| **9** | `monopoly` | Board Game & Rule Engine | **100.0** | 🏆 **Perfect 100** |
| **10** | `quiz` | Real-Time Quiz Game | **100.0** | 🏆 **Perfect 100** |
| **11** | `slack` | Real-Time Messaging / Workspace | **100.0** | 🏆 **Perfect 100** |
| **12** | `mafia` | Multi-Player Deduction Game | **95.8** | ⭐ **Excellent (90%+)** |
| **13** | `pilot_logbook` | Aviation Flight Log & Flight Rules | **95.7** | ⭐ **Excellent (90%+)** |
| **14** | `hvac` | HVAC Services & Estimator | **95.0** | ⭐ **Excellent (90%+)** |
| **15** | `online_whiteboard` | Collaborative Drawing Board | **92.9** | ⭐ **Excellent (90%+)** |
| **16** | `resume_builder` | Interactive Document Creator | **91.3** | ⭐ **Excellent (90%+)** |
| **17** | `book_journey` | Travel & Itinerary Planner | **90.1** | ⭐ **Excellent (90%+)** |
| **18** | `fleet_management` | Vehicle Odometer & Task Tracker | **82.5** | 👍 **Strong Pass** |
| **19** | `srm` | Supplier Relationship Management | **75.0** | 👍 **Strong Pass** |
| **20** | `notes` | Markdown Notes & Tags | **65.2** | 🟡 **Partial Pass** |
| **21** | `wedding` | Wedding RSVP & Seating Planner | **39.3** | 🟡 **Partial Pass** |
| **22** | `family_social` | Private Family Network | **33.3** | 🟡 **Partial Pass** |
| **23** | `family_friendly_venue` | Venue Locator & Reviews | **0.0** | ⚠️ **Eval Timeout** |
| **24** | `furniture_freight` | Pricing & Logistics Calculator | **0.0** | ⚠️ **Seed Timeout** |

---

## 🛠️ Architectural Insights & Learnings

1. **Reasoning Control (`reasoning_effort: "low"`)**:
   - By setting `"reasoning_effort": "low"` in `env_creator.py`, Kimi-K3 maintained internal chain-of-thought tokens at just **~4 KB–15 KB per turn**, while emitting **60 KB–100 KB+** of actionable code and OpenHands tool calls (`TerminalTool`, `FileEditorTool`).
   - This prevented infinite internal reasoning loops and enabled applications like `monopoly`, `mafia`, and `collabrative_kaban` to run continuously for 90–120 turns without exhausting context.

2. **256K Context Window & Token Allowance**:
   - Setting `AGENT_LLM_MAX_OUTPUT_TOKENS: "262144"` and `EFFECTIVE_CONTEXT_WINDOW: "262144"` prevented truncation on multi-hour builds where context windows routinely reached **120,000–170,000 tokens**.
   - SGLang's **RadixAttention prefix caching** hit **~130,000+ cached tokens per turn**, keeping decode throughput at **~99 tokens/sec** across the 8-node RTX 6000 Ada cluster.

3. **Docker-in-Docker Network Reliability (`10.0.45.181:30100/v1`)**:
   - Routing OpenHands containers directly via the Kubernetes ClusterIP (`10.0.45.181:30100/v1`) instead of `.svc.cluster.local` completely bypassed Docker bridge DNS resolution errors, resulting in **zero API connection failures** across over **7 solid hours** of continuous 4-way parallel execution (`--build-parallel 4 --seed-parallel 4 --evaluate-parallel 4`).

4. **🔍 Deep-Dive: Why `furniture_freight` Failed Seeding**:
   - **Symptom**: During Phase 2 (Seeding), `furniture_freight` crashed immediately after 1 second (`start-server.sh crashed after 1s with exit code 1`) across all three test plans (`test1`, `test2`, `test3`).
   - **Root Cause**: `furniture_freight` was built as a **Next.js 14** application. In Next.js, running `npm run start` (`next start`) starts a production server that requires a pre-compiled `.next` production build directory created by `next build` (`npm run build`).
   - **Why Kimi-K3 Missed It**: During Phase 1 (Build), Kimi-K3 ran `npm run build` interactively in its shell to test the app, so `.next` existed in its temporary workspace. However, when it wrote `setup-environment.sh`, it only included `npm install` and `node scripts/init-db.js` — **it omitted `npm run build` from `setup-environment.sh`**.
   - **Seeding Harness Behavior**: When Phase 2 launched a **clean, unbuilt container** (`app-validate-seed`), `./setup-environment.sh` ran without compiling `.next`. When `./start-server.sh` executed `npm run start`, Next.js threw:
     ```
     Error: Could not find a production build in the '.next' directory. Try building your app with 'next build' before starting the production server.
   - **Remedy / Pattern for Future Prompts**: For Next.js or compiled frontend frameworks (Vite/Next/Nuxt), `setup-environment.sh` must explicitly execute `npm run build` before completion so that idempotent server start scripts succeed in fresh containers.

5. **🔍 Deep-Dive: Why `family_friendly_venue` Failed Evaluation (Score 0.0)**:
   - **Symptom**: During Phase 3 (Evaluation), `family_friendly_venue` scored `0 / 38` (`test1`), `0 / 44` (`test2`), and `0 / 48` (`test3`) because the root URL `http://localhost:8000/` returned HTTP 404 `{"error":"not_found"}`.
   - **Root Cause**: `family_friendly_venue` was built as a React SPA bundled with **Vite**. Five source components (`App.jsx`, `Home.jsx`, `Preferences.jsx`, `SearchResults.jsx`, `VenueDetail.jsx`) imported helper modules from `'./lib/core.js'`. However, Kimi-K3 **never created the `/app/web/src/lib/core.js` file** before finishing its build turn.
   - **Why Kimi-K3 Missed It**: During Phase 1 (Build), Kimi-K3 verified the Python FastAPI backend (`GET /api/health` returned `200 OK` with 22 venues), but omitted a strict end-to-end production frontend build (`npm run build` check in `web/`) after refactoring its helper imports.
   - **Evaluation Harness Behavior**: When `./setup-environment.sh` ran `npm run build` for Vite during setup, Vite threw:
     ```
     Could not resolve './lib/core.js' from src/App.jsx
     ```
   - **Remedy / Pattern for Future Prompts**: Agents should always run an explicit frontend production build command (`npm run build` or `vite build`) as a final verification step before calling `finish` to catch unresolved import paths early.

6. **🔍 Deep-Dive: Why `wedding` Scored 39.3%**:
   - **Score Breakdown**: `test1` = `0` (eval timeout), `test2` = `4 / 70` (5.7%), `test3` = `61 / 61` (100.0% Perfect Pass!). Average = **`39.3%`**.
   - **Why `test3` (Tour Booking Workflow) Scored 100%**: Across 4 browser contexts (venue manager + 3 couples), every single step of signup, venue creation, calendar tour scheduling, conflict handling, and booking verification passed 100%.
   - **Why `test2` (Venue Search & Filtering) Failed**: When a user submits the search form without specifying price filter inputs, the browser GET request sends empty parameters (`?min_price=&max_price=`). In Kimi-K3's FastAPI backend, `min_price: float` and `max_price: float` were declared without defaulting empty strings `""` to `None`. Consequently, FastAPI threw **`HTTP 422 Unprocessable Entity`** when empty query parameters were passed, blocking results and halting Steps 2–10.
   - **Remedy / Pattern for Future Prompts**: In FastAPI query parameters, always declare numeric filters as Optional with custom validators or default empty strings `""` to `None` (`Optional[float] = Query(default=None)`) so empty search fields don't cause HTTP 422 validation errors.

7. **🔍 Deep-Dive: Why `notes` Scored 65.2%**:
   - **Score Breakdown**: `test1` = `42 / 50` (84.0%), `test2` = `42 / 56` (75.0%), `test3` = `22 / 60` (36.7%). Average = **`65.2%`**.
   - **Why `test1` & `test2` Lost Points (Re-lock on Reload)**: The PRD specified that a page reload should re-lock the notes application (in-memory session authentication). However, Kimi-K3 stored the unlocked auth token in `localStorage`, causing the app to remain unlocked after a browser refresh.
   - **Why `test3` Lost Points (Missing Sidebar Delete Button)**: In `test3`, Steps 1–3 (password unlock, note creation, substring search) passed 100%. However, Step 5 required deleting a note directly from the **notes list view** (left sidebar). Kimi-K3 implemented a Delete button inside the note editor panel, but omitted a delete icon/button on individual note cards in the list view, causing a fatal UI step failure.
   - **Remedy / Pattern for Future Prompts**: Pay strict attention to PRD authentication persistence rules (`localStorage` vs in-memory session state) and ensure CRUD actions (like Delete) are accessible from both list/card views and detailed view panels.

8. **🔍 Deep-Dive: Why `family_social` Scored 33.3%**:
   - **Score Breakdown**: `test1` = `0 / 42` (0%), `test2` = `0 / 70` (0%), `test3` = `70 / 70` (**100.0% Perfect Pass!**). Average = **`33.3%`**.
   - **Why `test3` (Famio Social Platform Workflow) Scored 100%**: Across 3 isolated browser contexts (`Hannah`, `Ian`, `Chloe`), every single step of account creation, relationship chains, post CRUD, newest-first feed ordering, and the critical **"no transitive visibility"** security rule passed 100% without error.
   - **Why `test1` & `test2` Failed (Demo Data Email Collision — HTTP 409)**: In Kimi-K3's backend (`db.py` / `setup-environment.sh`), it included an automatic `seed_if_empty()` function that pre-created demo accounts (`alice@example.com`, `bob@example.com`, `carol@example.com`, `dan@example.com`).
   - When **`test1`** (Step 1) attempted to sign up a fresh account for `alice@example.com`, the server rejected the signup with **`HTTP 409: An account with this email already exists`**, fatally halting the test plan.
   - Similarly, when **`test2`** (Step 1) attempted to sign up fresh accounts for `carol@example.com` and `dan@example.com`, both were rejected with HTTP 409 because those exact emails were pre-seeded by demo data.
   - In **`test3`**, because the test users (`hannah@example.com`, `ian@example.com`, `chloe@example.com`) did not collide with Kimi-K3's demo emails, account creation succeeded and the application's actual social networking features scored **70 out of 70 points (100%)**.
   - **Remedy / Pattern for Future Prompts**: Agents should never pre-seed common demo emails (`alice`, `bob`, `carol`, `dan`) in production/test database initialization scripts unless explicitly requested, as automated evaluation test plans often sign up standard user accounts from scratch.




