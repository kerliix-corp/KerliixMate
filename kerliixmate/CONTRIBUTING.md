# Contributing to the Kerliix Website

Welcome to the **Kerliix Internal Web Platform** repository.  
This documentation outlines the expectations, standards, and procedures for contributing to the Kerliix website.

> **Note**: This is a **private, internal-use-only** repository. All contributions are restricted to verified members of the Kerliix Corporation engineering team.

---

## Internal Contributions Only

This is **not** an open-source project. Contributions from individuals or entities outside of the Kerliix Corporation are **not accepted**.

- Do **not** submit issues, pull requests, or feature suggestions unless you are a Kerliix team member.
- Do **not** fork or share this repository externally under any circumstance.
- Access is restricted to approved personnel with proper credentials and NDA compliance.

---

## Contribution Requirements

All contributions must go through internal planning and adhere to our product, security, and design standards.

### Planning & Tasks

- All work must originate from a task in our internal issue tracking system (e.g., Jira).
- Only begin a task that has been assigned or approved by a team lead or project manager.

### Development Standards

- Follow Kerliix’s internal **coding standards** and **best practices** for the tech stack in use (e.g., TypeScript, React, SCSS, etc.).
- Maintain **clean, readable**, and **well-documented** code.
- Include relevant **unit tests**, **integration tests**, and **visual regression testing** if applicable.
- Always perform **local testing** before opening a pull request.

### Commit Message Format

Use clear and consistent commit messages. Follow this format:

[type]: short description

_Optional body explaining what and why._

### Examples:

- `feat: add new contact form to footer`
- `fix: correct alignment issue in hero banner on mobile`
- `chore: update dependencies and fix minor linter warnings`

---

### Code Review Process

- All changes must be submitted via Pull Request (PR) and pass all automated checks.
- At least **one senior developer or team lead** must approve the PR before merging.
- All design-related updates must be reviewed by the **design owner or UX lead**.

---

## Branching Strategy

We follow a strict branching model:

- `main` — Production-ready code (auto-deployed)
- `develop` — Staging integration branch
- `feature/*` — New features (branch from `develop`)
- `bugfix/*` — Bug fixes (branch from `develop`)
- `hotfix/*` — Critical production fixes (branch from `main`, then merged back)

> ⚠️ **Never commit directly to `main` or `live`.** All changes must be made through branches and pull requests.

---

## Repository Confidentiality

This repository and all of its contents are **confidential and proprietary** to **Kerliix Corporation**.

- Do **not** share, clone, mirror, or replicate this project to external systems.
- All development must take place within the approved Kerliix infrastructure.
- Violations of these terms may result in disciplinary action and legal consequences.

---

## Questions or Issues?

For any questions regarding contributions, process, or access:

- Reach out to your **team lead**.
- Contact the **Engineering Manager** or **Project Manager**.
- Refer to our internal Confluence documentation for technical and process references.

---

> — **Kerliix Engineering Team**
