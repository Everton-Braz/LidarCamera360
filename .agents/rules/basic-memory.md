---
trigger: always_on
description: Query basic-memory and graphify at session start, and save decisions to basic-memory.
---

# Persistent Memory & Knowledge Graph Rule

## Session Start Workflow
1. **Query Graphify Knowledge Graph**:
   - For codebase and architecture questions, consult `graphify-out/graph.json` via `graphify query "<question>"`, `graphify path "<A>" "<B>"`, or `graphify explain "<concept>"`.
   - The interactive graph visualizer is available at `graphify-out/graph.html`.
2. **Check Basic-Memory**:
   - Query `basic-memory` MCP server via `recent_activity` and `search_notes` on project `lidar-camera-calibrator` to check recent decisions, calibration parameters, and structural choices before editing code.
   - Project location: `C:/Users/Everton-PC/Documents/ObsidianVault/projects/Lidar-camera-calibrator`.

## During & After Code Modifications
1. **Record Important Decisions**:
   - Save important architectural, mathematical, or configuration decisions to `basic-memory` using `write_note` (project: `lidar-camera-calibrator`, directories: `decisions/`, `calibration/`, or `architecture/`).
2. **Keep Knowledge Graph Current**:
   - Run `graphify update .` after code changes to incrementally refresh `graphify-out/graph.json`.
3. **Obsidian Sync**:
   - Obsidian notes and `graph.canvas` are located in `C:/Users/Everton-PC/Documents/ObsidianVault/projects/Lidar-camera-calibrator`.
