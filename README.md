# GitHub Pages Website - Gwen Mabon

Personal site built with Hugo and the [Hextra](https://imfing.github.io/hextra/) theme: math-first AdTech/ML field notes, a short recruiter-facing about page, and an archive of academic research.

## Structure

- `config/_default/` - Hugo configuration (`hugo.yaml`, `menus.yaml`, `params.yaml`, `module.yaml`)
- `content/` - Site content
  - `_index.md` - Homepage
  - `field-notes/` - Math-first AdTech/ML notes, chaptered (sidebar navigation, KaTeX math, syntax highlighting)
  - `about/` - Bio and contact
  - `research/` - Academic publications, thesis, talks (from PhD, 2013-2017)
- `assets/css/custom.css` - Theme color/font overrides
- Theme is pulled in via Hugo Modules (`go.mod` / `config/_default/module.yaml`), no theme submodule needed

## Installation

1. **Install Hugo Extended**: [Hugo Documentation](https://gohugo.io/getting-started/installing/)

2. **Download Hugo Modules**:
   ```bash
   hugo mod tidy
   ```

3. **Preview the site locally**:
   ```bash
   hugo server -D
   ```
   Visit http://localhost:1313

4. **Build the site**:
   ```bash
   hugo --minify
   ```

5. **Deploy to GitHub Pages**: handled automatically by `.github/workflows/hugo.yml` on push to `main`.

## Customization

- Add new field notes under `content/field-notes/<topic-name>/`, ordered with the `weight` front matter field.
- Colors and fonts: `assets/css/custom.css` (see [Hextra customization docs](https://imfing.github.io/hextra/docs/advanced/customization/)).
- Navigation: `config/_default/menus.yaml`.
