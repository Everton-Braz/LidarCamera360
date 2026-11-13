---
name: release
description: >-
  Ship a LidarCamera360 release end to end: check readiness, build and tag, publish the GitHub release with
  binaries, then make the promo kit (real UI captures, banners, fframes release film) and the copy for
  LinkedIn, Facebook and X with a step-by-step posting checklist. Use when the user says "release", "lançar",
  "publicar versão", "post do release", "banner", "vídeo do release", or names a version like v0.5.0.
---

# Release LidarCamera360

Work in this order. Each phase ends with something the user can check. Never publish, push or post
without the user's explicit yes for that step.

**Hard rule:** this app is free and stands alone. No text, image, video or post may mention any other product,
a paid edition, "Pro", "Lite" or "coming soon" features of another app.

## 1. Readiness

1. `git log <prev-tag>..dev --oneline` and `CHANGELOG.md`: every user-facing change is in the changelog.
2. Version strings agree: `raven_app/__init__.py`, `native/CMakeLists.txt`, `native/main.cpp`, `README.md`.
3. Tests: see the memory note on Python envs (Qt and pyproj live in different interpreters; run per file).
4. Release notes in `docs/releases/<version>.md`: English, short sections, "Known limitations", "Binaries".
   Only claim what was tested on a real dataset. Ask the user when a feature is untested.

## 2. Build, tag, publish

```bash
git checkout main && git merge --ff-only dev
git tag -a <version> -m "LidarCamera360 <version>"
```

Build the zip and the portable exe into `dist/`, smoke-test both (launch, open a dataset, close), record sha256:

```bash
certutil -hashfile dist/LidarCamera360-<version>-windows-x64.zip SHA256
```

**Ask before this block.** It is public and hard to undo.

```bash
git push origin main dev <version>
gh release create <version> dist/LidarCamera360-<version>-windows-x64.zip dist/LidarCamera360_portable.exe --title "LidarCamera360 <version>" --notes-file docs/releases/<version>.md
```

Then `gh release view <version> --web` and check: assets download, notes render, the tag points at main.
Attach the banner and the film to the release body afterwards if wanted (drag into the web editor).

## 3. Promo kit (`promo/<version>/`)

| Asset | Tool | Output |
|---|---|---|
| UI captures | `Python311 promo/tools/capture_ui.py promo/<v>/captures --theme dark` | `captures/dark_0N_<page>.png` 3200x2000 |
| Banners | `python promo/tools/make_banner.py --version <v> --config promo/<v>/banner.json` | `banner/banner_1920x1080.jpg`, `_1200x675`, `_1080x1350` |
| Film | fframes project in `promo/<v>/film/` (copy the previous version's, minus `target/`) | `film/release_<v>.mp4` |
| Copy | `promo/<v>/posts.md` from [references/post-templates.md](references/post-templates.md) | one block per platform |

- Captures use Python311 (has PyQt6 + qfluentwidgets). They disable Mica and grab the screen, so keep the window
  unobstructed while it runs.
- `banner.json`: headline (2 lines, ≤ 24 chars each), subline, ≤ 6 chips, the UI crop, optional `hero`
  (a real 3DGS or point-cloud render, 1920x1080). Look at every output size before using it.
- The film: real UI only, punched in so text reads at ≥ 26 px on screen; real 3DGS / cloud footage from the user
  goes in `film/footage/` under the names listed in `film/README.md`. Logo in the last 10%, CTA held ≥ 1.5 s.
- The film is made with **fframes** (https://fframes.studio, Rust + SVG, MIT). Agent skill:
  `npx skills add https://fframes.studio`. Windows setup: `cargo install --locked cargo-fframes`, LLVM
  (`winget install LLVM.LLVM`, `LIBCLANG_PATH`), an FFmpeg 9.0 *shared* build (`FFMPEG_DIR`, its `bin` on `PATH`),
  codec features off. Keep `fframes-media`, `fframes-media-dir-macro` and `svgr-macro` at the fframes version
  (`cargo update -p <crate> --precise 1.4.0`): newer ones pull a second `usvgr` and the build fails with E0308.
  First build compiles Skia (~20 min). Ask the user before installing any of it.
  Loop: `cargo run --release -- inspect`, `strip`, `frame`, then `preview` for the user, then `render`.
- Ask the user for footage early: a 3DGS fly-through (≥ 4 s) and a before/after cloud pair from one viewpoint.

## 4. Posting

Follow [references/social-platforms.md](references/social-platforms.md): specs, limits and click-by-click steps
for LinkedIn, Facebook and X. Order: GitHub release live → X → LinkedIn → Facebook (same day, morning).
The user posts; Claude drafts and checks. Sending a post on the user's behalf needs their yes per post.

After posting: put the post links in `promo/<v>/posts.md` under "Published", reply to early comments within
the first hour (the first hour decides reach on all three).
