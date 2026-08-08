# ViBench Hard Benchmark Final Report: Kimi-K3 on SGLang (GKE GB200)

> **Execution Date**: Saturday, August 8, 2026
> **Model Under Test**: `moonshotai/Kimi-K3` (SGLang Multi-Host serving with MNNVL + DCP enabled across 8 Nodes / 32 GPUs)
> **Evaluation Environment**: ViBench Brownfield Feature Extension Suite (4-Way Parallel OpenHands Agents & Playwright Headless Browser)

---

## 🏆 Executive Summary

| Metric | Value | Benchmark Context |
| :--- | :---: | :--- |
| **Overall Normalized Score** | **79.2 / 100** | Primary benchmark ranking metric |
| **Zero-Score (0%) Artifacts** | **0 (0.0%)** | 100% of feature extensions produced working code |
| **Perfect Pass (100%) Artifacts** | **9 (45.0%)** | 9 out of 20 feature artifacts passed all assertions flawlessly |
| **Total Test Plans Evaluated** | **63 plans** | Full end-to-end Playwright browser assertion runs |
| **Average Feature Build Time** | **44.7 min** | Code synthesis + automated sandbox unit testing |
| **Cost / Token Incurred** | **$0.00** | Self-hosted GKE inference |

---

## 📊 Performance by Feature Complexity Tier

| Complexity Tier | Total Test Plans | Normalized Avg Score | 100% Pass Rate |
| :--- | :---: | :---: | :---: |
| **Feature 1** *(Initial extension & schema additions)* | 26 | **64.2 / 100** | 50.0% (13/26) |
| **Feature 2** *(Intermediate business logic & workflows)* | 17 | **83.9 / 100** | 64.7% (11/17) |
| **Feature 3** *(Advanced full-stack integrations)* | 17 | **84.8 / 100** | 70.6% (12/17) |
| **Feature 4** *(Complex brownfield refactoring)* | 3 | **96.8 / 100** | 66.7% (2/3) |

---

## 🏗️ Detailed Project Scorecard

| Project Application | Evaluated Plans | Normalized Score | Pass Rate | Key Performance Characteristics |
| :--- | :---: | :---: | :---: | :--- |
| **`collabrative_kaban`** | 3 | **100.0 / 100** | 100.0% (3/3) | 🌟 Flawless implementation of kanban board feature extension |
| **`market_place`** | 6 | **98.4 / 100** | 83.3% (5/6) | Excellent e-commerce order workflow & catalog extension |
| **`monopoly`** | 3 | **93.9 / 100** | 66.7% (2/3) | Complex multi-player board game rules & state transitions |
| **`family_friendly_venue`** | 4 | **81.9 / 100** | 50.0% (2/4) | Venue geolocation & search filter extensions |
| **`furniture_freight`** | 11 | **81.8 / 100** | 81.8% (9/11) | High volume freight capacity estimation & quote booking |
| **`srm`** | 7 | **81.6 / 100** | 42.9% (3/7) | Supplier Relationship Management CRUD & contracts |
| **`online_whiteboard`** | 3 | **80.8 / 100** | 33.3% (1/3) | Real-time HTML5 canvas rendering & shape manipulation |
| **`hvac`** | 12 | **75.0 / 100** | 75.0% (9/12) | Sensor telemetry streaming & HVAC thermostat control |
| **`pilot_logbook`** | 7 | **68.3 / 100** | 57.1% (4/7) | FAA compliance flight hour tracking & currency calculations |
| **`slack`** | 7 | **33.8 / 100** | 0.0% (0/7) | Real-time multi-context synchronization lag & duplicate DOM state |

---

## 🔬 Deep Dive: Project-by-Project Test Step Assertions

### 📦 `collabrative_kaban`

| Feature | Test Plan | Score | Full Points | Normalized | Steps Passed | Complete Pass | Complete Fail |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `feature3` | `regression` | 30 | 30 | 100.0% | 5/5 | ✅ Yes | — |
| `feature3` | `test1` | 47 | 47 | 100.0% | 8/8 | ✅ Yes | — |
| `feature3` | `test2` | 64 | 64 | 100.0% | 7/7 | ✅ Yes | — |

### 📦 `family_friendly_venue`

| Feature | Test Plan | Score | Full Points | Normalized | Steps Passed | Complete Pass | Complete Fail |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `feature1` | `regression` | 40 | 40 | 100.0% | 7/7 | ✅ Yes | — |
| `feature1` | `test1` | 32 | 58 | 55.2% | 4/8 | ❌ No | — |
| `feature1` | `test2` | 42 | 58 | 72.4% | 6/9 | ❌ No | — |
| `feature1` | `test3` | 38 | 38 | 100.0% | 6/6 | ✅ Yes | — |

### 📦 `furniture_freight`

| Feature | Test Plan | Score | Full Points | Normalized | Steps Passed | Complete Pass | Complete Fail |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `feature1` | `regression` | 10 | 10 | 100.0% | 2/2 | ✅ Yes | — |
| `feature1` | `test1` | 0 | 25 | 0.0% | 0/3 | ❌ No | 🔴 Fail |
| `feature1` | `test2` | 25 | 25 | 100.0% | 3/3 | ✅ Yes | — |
| `feature1` | `test3` | 0 | 45 | 0.0% | 0/6 | ❌ No | 🔴 Fail |
| `feature2` | `regression` | 15 | 15 | 100.0% | 2/2 | ✅ Yes | — |
| `feature2` | `test1` | 24 | 24 | 100.0% | 3/3 | ✅ Yes | — |
| `feature2` | `test2` | 26 | 26 | 100.0% | 3/3 | ✅ Yes | — |
| `feature3` | `regression` | 20 | 20 | 100.0% | 2/2 | ✅ Yes | — |
| `feature3` | `test1` | 27 | 27 | 100.0% | 4/4 | ✅ Yes | — |
| `feature3` | `test2` | 40 | 40 | 100.0% | 5/5 | ✅ Yes | — |
| `feature3` | `test3` | 27 | 27 | 100.0% | 3/3 | ✅ Yes | — |

### 📦 `hvac`

| Feature | Test Plan | Score | Full Points | Normalized | Steps Passed | Complete Pass | Complete Fail |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `feature1` | `regression` | 14 | 14 | 100.0% | 5/5 | ✅ Yes | — |
| `feature1` | `test1` | 0 | 19 | 0.0% | 0/6 | ❌ No | 🔴 Fail |
| `feature1` | `test2` | 0 | 19 | 0.0% | 0/8 | ❌ No | 🔴 Fail |
| `feature1` | `test3` | 0 | 25 | 0.0% | 0/8 | ❌ No | 🔴 Fail |
| `feature2` | `regression` | 70 | 70 | 100.0% | 6/6 | ✅ Yes | — |
| `feature2` | `test1` | 45 | 45 | 100.0% | 3/3 | ✅ Yes | — |
| `feature2` | `test2` | 90 | 90 | 100.0% | 6/6 | ✅ Yes | — |
| `feature2` | `test3` | 90 | 90 | 100.0% | 5/5 | ✅ Yes | — |
| `feature3` | `regression` | 40 | 40 | 100.0% | 4/4 | ✅ Yes | — |
| `feature3` | `test1` | 35 | 35 | 100.0% | 3/3 | ✅ Yes | — |
| `feature3` | `test2` | 60 | 60 | 100.0% | 6/6 | ✅ Yes | — |
| `feature3` | `test3` | 48 | 48 | 100.0% | 5/5 | ✅ Yes | — |

### 📦 `market_place`

| Feature | Test Plan | Score | Full Points | Normalized | Steps Passed | Complete Pass | Complete Fail |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `feature1` | `test1` | 37 | 37 | 100.0% | 6/6 | ✅ Yes | — |
| `feature1` | `test2` | 36 | 36 | 100.0% | 6/6 | ✅ Yes | — |
| `feature1` | `test3` | 31 | 31 | 100.0% | 6/6 | ✅ Yes | — |
| `feature4` | `regression` | 29 | 29 | 100.0% | 6/6 | ✅ Yes | — |
| `feature4` | `test1` | 46 | 46 | 100.0% | 8/8 | ✅ Yes | — |
| `feature4` | `test2` | 38 | 42 | 90.5% | 6/7 | ❌ No | — |

### 📦 `monopoly`

| Feature | Test Plan | Score | Full Points | Normalized | Steps Passed | Complete Pass | Complete Fail |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `feature2` | `regression` | 49 | 60 | 81.7% | 8/9 | ❌ No | — |
| `feature2` | `test1` | 70 | 70 | 100.0% | 9/9 | ✅ Yes | — |
| `feature2` | `test2` | 70 | 70 | 100.0% | 13/13 | ✅ Yes | — |

### 📦 `online_whiteboard`

| Feature | Test Plan | Score | Full Points | Normalized | Steps Passed | Complete Pass | Complete Fail |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `feature1` | `regression` | 40 | 40 | 100.0% | 4/4 | ✅ Yes | — |
| `feature1` | `test1` | 42 | 52 | 80.8% | 4/5 | ❌ No | — |
| `feature1` | `test2` | 32 | 52 | 61.5% | 3/5 | ❌ No | — |

### 📦 `pilot_logbook`

| Feature | Test Plan | Score | Full Points | Normalized | Steps Passed | Complete Pass | Complete Fail |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `feature1` | `regression` | 43 | 43 | 100.0% | 7/7 | ✅ Yes | — |
| `feature1` | `test1` | 68 | 68 | 100.0% | 10/10 | ✅ Yes | — |
| `feature2` | `regression` | 52 | 52 | 100.0% | 7/7 | ✅ Yes | — |
| `feature2` | `test1` | 26 | 54 | 48.1% | 4/7 | ❌ No | — |
| `feature2` | `test2` | 16 | 54 | 29.6% | 2/6 | ❌ No | — |
| `feature2` | `test3` | 0 | 94 | 0.0% | 0/12 | ❌ No | 🔴 Fail |
| `feature3` | `regression` | 35 | 35 | 100.0% | 5/5 | ✅ Yes | — |

### 📦 `slack`

| Feature | Test Plan | Score | Full Points | Normalized | Steps Passed | Complete Pass | Complete Fail |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `feature1` | `regression` | 14 | 40 | 35.0% | 2/5 | ❌ No | — |
| `feature1` | `test1` | 44 | 80 | 55.0% | 5/8 | ❌ No | — |
| `feature1` | `test2` | 7 | 76 | 9.2% | 2/9 | ❌ No | — |
| `feature1` | `test3` | 0 | 66 | 0.0% | 0/8 | ❌ No | 🔴 Fail |
| `feature3` | `regression` | 8 | 40 | 20.0% | 1/4 | ❌ No | — |
| `feature3` | `test1` | 34 | 48 | 70.8% | 4/6 | ❌ No | — |
| `feature3` | `test2` | 31 | 67 | 46.3% | 4/8 | ❌ No | — |

### 📦 `srm`

| Feature | Test Plan | Score | Full Points | Normalized | Steps Passed | Complete Pass | Complete Fail |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `feature1` | `test1` | 68 | 68 | 100.0% | 10/10 | ✅ Yes | — |
| `feature1` | `test2` | 64 | 64 | 100.0% | 10/10 | ✅ Yes | — |
| `feature2` | `test1` | 70 | 70 | 100.0% | 9/9 | ✅ Yes | — |
| `feature2` | `test3` | 58 | 66 | 87.9% | 5/6 | ❌ No | — |
| `feature2` | `test4` | 45 | 57 | 79.0% | 5/6 | ❌ No | — |
| `feature3` | `regression` | 13 | 79 | 16.5% | 2/14 | ❌ No | — |
| `feature3` | `test2` | 44 | 50 | 88.0% | 5/6 | ❌ No | — |
