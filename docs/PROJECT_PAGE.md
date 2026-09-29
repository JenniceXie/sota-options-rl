# Project page and figures

The static project page is `docs/index.html`. It uses local SVG, CSS, and
JavaScript assets; there are no third-party scripts, analytics, or build-time
JavaScript dependencies.

## Preview

From the repository root:

```sh
python -m http.server 8000 --directory docs
```

Open `http://localhost:8000`. Opening `docs/index.html` directly also works.

## Update

`sota_framework.svg` is the approved Variant A. The previous JPEG is retained
for reference. `sota_banner.svg` is the editable project headline; the PNG
versions provide raster fallbacks.

After changing the framework SVG or `results/table1.csv`, regenerate the page:

```sh
python scripts/build_project_page.py
```

The page builder uses the Python standard library. If the SVG layout changes,
update its node/flow annotations in the builder. Explanatory step text is in
`docs/site.js`.

## Animation and accessibility

The walkthrough starts paused. Readers can select any step, use previous/next,
or play one cycle. Left/right arrow keys navigate when a walkthrough button is
focused. Playback pauses when the page is hidden. Reduced-motion preferences
disable animated arrow dashes and transitions. Without JavaScript, the full
static diagram and its explanation remain available.

## Hosting

This is ready for GitHub Pages using **Deploy from a branch**, branch `main`,
folder `/docs`. No deployment is performed by the page builder.

The colors and rounded-card treatment draw on the STORM project-page style.
The SOTA illustrations and implementation were created for this repository.
