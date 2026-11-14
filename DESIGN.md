---
version: alpha
name: FaceAttend
description: Workforce attendance and employee operations for organisation teams and system administrators.
colors:
  primary: "#667eea"
  primary-dark: "#5a67d8"
  secondary: "#764ba2"
  accent: "#f093fb"
  text: "#1e293b"
  muted: "#64748b"
  background: "#f8fafc"
  surface: "#ffffff"
  organisation-teal: "#087f82"
  organisation-navy: "#102a36"
typography:
  display:
    fontFamily: "Outfit, system-ui, sans-serif"
  sans:
    fontFamily: "Inter, system-ui, sans-serif"
  mono:
    fontFamily: "ui-monospace, SFMono-Regular, Consolas, monospace"
rounded:
  sm: "8px"
  DEFAULT: "12px"
  lg: "16px"
  xl: "20px"
spacing:
  content-padding: "2rem"
  section-gap: "1.25rem"
  header-height: "72px"
components:
  notification-menu:
    width: "360px maximum; viewport minus 24px on narrow screens"
    unread: "Pale blue-green surface with text label and count badge"
  audit-event:
    surface: "White card with a thin neutral border and a teal timeline marker"
    metadata: "Employee, employee code, organisation, and exact event time"
  organisation-chart-card:
    width: "260px desktop; 238px narrow viewport"
    connector: "Muted blue-grey orthogonal tree lines"
    current-employee: "Teal border plus a visible You label"
---

# FaceAttend Design System

## Overview

### Creative North Star

FaceAttend is an operations console for workforce records. Its visual language takes cues from a well-maintained attendance register and a clearly labelled building directory: dates and identities stay easy to scan, while hierarchy is shown with explicit labels and thin connectors.

### Product context and register

- **Audience and primary job:** Employees, organisation HR and administrators, and platform superusers manage attendance and workforce records.
- **Target market and locale evidence:** The current interface is English (`en-us`) and the application timezone defaults to `Asia/Kolkata`; the repository does not define a locale provider or additional supported markets.
- **Usage scene:** A desktop operations workspace with employee self-service on phones. Tables and charts must remain usable with touch, keyboard, and narrow viewports.
- **Register:** Product UI. Task clarity, tenant boundaries, and readable records take priority over decorative effects.
- **Memorable signature:** The organisation chart uses continuous, orthogonal connectors so reporting relationships remain visible across branches.
- **Restraint:** Use gradients sparingly. Keep record cards, notification states, and audit metadata quiet and legible.
- **Anti-references:** Avoid promotional hero layouts, decorative metric cards, and color-only indicators; those patterns obscure the operational purpose.
- **Token ownership/runtime mapping:** `templates/base.html` owns global font and color variables. `DESIGN.md` mirrors those values. Audit and organisation chart components use page-scoped tokens in their templates.

## Colors

The global shell uses indigo `#667eea` and violet `#764ba2` as brand accents, with slate `#1e293b` text over white and `#f8fafc` surfaces. The organisation chart and audit timeline use teal `#087f82` and deep blue-green `#102a36` to identify hierarchy and system records. Reserve red for unread counts or destructive/error states. Keep labels alongside color states and retain a visible focus outline.

## Typography

Outfit is used for page headings and Inter for body text, matching the global shell. Use system sans-serif fallbacks so pages remain usable if the font CDN is unavailable. Use tabular numerals for dates and counts. Keep employee codes and request IDs readable and wrap them when necessary; do not truncate the only copy of an identifier.

## Layout

The shared content area uses 2rem desktop padding and a 72px header. Feature pages may reduce padding at phone widths and must reflow toolbars before shrinking text. The chart has its own two-axis scroll region; the rest of the page retains document scrolling. Keep page titles, filters, and primary actions visible before dense records.

## Elevation & Depth

Use borders and pale surface changes to separate operational regions. Cards may use a small diffuse shadow to lift them from the page; avoid glow and blur as a default. Dropdowns sit above page content with a clear border and bounded scrolling.

## Shapes

Global controls and surfaces use 8px, 12px, 16px, and 20px radii. Organisation cards use a tighter 11px radius to keep the tree compact. Use thin dividers and subdued connector strokes; reserve filled pills for short status labels.

## Components

### Foundational visual states

Interactive controls have hover, active, disabled, and visible keyboard-focus states. Unread notifications combine a numeric badge with an `Unread` label. Empty and error states explain what is missing and what the user can do next. Respect reduced-motion preferences.

### Buttons and actions

Use Bootstrap button geometry and the shell's indigo brand color. Keep icon-only controls at least 30px in compact toolbars and provide an accessible name; important touch actions should approach 44px.

### Navigation and data display

Audit history is an immutable paginated timeline. Every item shows its timestamp, actor, employee subject when available, employee code, and organisation. The organisation chart is a nested reporting tree with horizontal and vertical scrolling, branch controls, and keyboard-adjustable zoom.

### Forms and overlays

Use native selects where operating-system popup styling is acceptable. Dropdowns stay within the viewport width and have bounded vertical scrolling. Notification counts describe the linked notification list; the count never substitutes for its messages.

### Iconography

Font Awesome 6 is the current icon family. Icons support a text label and never carry essential status by themselves.

### Motion

Motion is limited to brief hover feedback and branch expansion. Disable nonessential transitions when reduced motion is requested.

### Content and data visualization

Use sentence case and direct action labels. Show exact timestamps in audit history, and use explicit labels for employee name, employee code, and organisation.

## Do's and Don'ts

- **Do:** Keep employee and organisation context next to the record it describes.
- **Do:** Keep organisation scope enforced by server queries, independent of the visible selector.
- **Don't:** Use placeholder counts or notification messages that are not backed by records.
- **Don't:** hide chart branches or unread states without an accessible label.
