# GitHub Actions — workflows & one-time setup

Two workflows live here:

| Workflow | Trigger | What it does |
|---|---|---|
| [`ci.yml`](ci.yml) | push to `main`, every PR, manual | Lint (`ruff check` + `ruff format --check`) and `pytest` across Python 3.10–3.13. No secrets, works out of the box. |
| [`release.yml`](release.yml) | push a `v*` tag, manual | Builds the sdist + wheel, validates metadata with `twine`, and publishes to PyPI via **Trusted Publishing** (OIDC — no stored token). |

`ci.yml` needs no configuration. `release.yml` requires the one-time setup below.

---

## One-time setup for `release.yml` (PyPI Trusted Publishing)

Trusted Publishing lets GitHub Actions upload to PyPI using a short-lived OIDC token
minted per run, so there is **no long-lived API token** to store or rotate.

### 1. Register the project as a Trusted Publisher on PyPI

1. Sign in at <https://pypi.org> and go to **Account settings → Publishing**
   (<https://pypi.org/manage/account/publishing/>).
2. Under **Add a new pending publisher**, fill in:
   - **PyPI Project Name:** `hermes-ollama-native`
   - **Owner:** `hanzlamateen`
   - **Repository name:** `hermes-ollama-native`
   - **Workflow name:** `release.yml`
   - **Environment name:** `pypi`
3. Save. (A "pending" publisher is fine before the project's first upload — the first
   tagged release will create the project and bind the publisher.)

> Want to dry-run first? Repeat the same steps at <https://test.pypi.org> and add a
> TestPyPI publish step (see the optional section below) before going live.

### 2. Create the `pypi` GitHub Environment

In the repo: **Settings → Environments → New environment → `pypi`**. Recommended
protections:

- **Deployment branches and tags:** restrict to tags matching `v*` so only tagged
  releases can publish.
- (Optional) **Required reviewers:** add yourself to gate each publish behind a manual
  approval click.

The environment name **must** match `release.yml`'s `environment.name: pypi` and the
publisher you registered on PyPI.

### 3. Confirm repo permissions

`release.yml` already declares the only permission Trusted Publishing needs:

```yaml
permissions:
  id-token: write   # mint the OIDC token used to authenticate to PyPI
```

No `PYPI_API_TOKEN` secret is required or used. Also ensure **Settings → Actions →
General → Workflow permissions** allows workflows to run (default for this repo).

### 4. Cut a release

```bash
# bump the version in pyproject.toml and add a CHANGELOG entry first, then:
git tag v0.1.0
git push origin v0.1.0
```

The tag push triggers `release.yml` → build → (approval, if you enabled it) → publish.
Watch progress under the repo's **Actions** tab. After it succeeds, the release is live
at <https://pypi.org/project/hermes-ollama-native/>.

---

## Optional: TestPyPI dry-run

To validate end-to-end before publishing to real PyPI, register a publisher on
<https://test.pypi.org> (step 1 against the test instance), then add a step to the
`publish` job ahead of the real publish:

```yaml
      - name: Publish to TestPyPI
        uses: pypa/gh-action-pypi-publish@release/v1
        with:
          repository-url: https://test.pypi.org/legacy/
```

Install from TestPyPI to verify:

```bash
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ hermes-ollama-native
```

---

## Checklist

- [ ] PyPI Trusted Publisher added (project `hermes-ollama-native`, repo
      `hanzlamateen/hermes-ollama-native`, workflow `release.yml`, environment `pypi`).
- [ ] GitHub Environment `pypi` created (tag-restricted; optional required reviewers).
- [ ] `release.yml` keeps `permissions: id-token: write` on the `publish` job.
- [ ] Version bumped in `pyproject.toml` + `CHANGELOG.md` updated.
- [ ] Tag pushed (`vX.Y.Z`); Actions run is green; package visible on PyPI.
