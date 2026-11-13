# Face Attendance — security and multi-organisation upgrade

This ZIP retains the original Django core applications, migrations, templates, employee management, leave, payroll and reports. It hardens their deployment settings and critical attendance workflow. **It has not been certified or load-tested as production secure.** Do not deploy onto real employee biometric records without staging tests, consent/privacy review and a validated PAD (presentation-attack detection) solution.

## Highlights

- Django 5.2.18 security baseline; secure cookies, HTTPS redirect, HSTS (production), fixed CORS allowlist, CSRF, CSP, no plaintext secrets committed.
- PostgreSQL in production; Redis-backed shared login/scan/API rate limiting and cross-worker scan votes; verified per-employee DB row locking and cooldown.
- Tenant-scoped query managers, explicit organisation context, active-org checks, safer finance model manager and global admin restricted to platform superusers.
- Encrypted face embeddings via Fernet, one-time schema/data migration, tenant-specific gallery matching, single-face/ambiguous-match rejection and bounded inference.
- Private image/document routing, decoded-image validation, WebP re-encoding of new uploads, static compressed+hashed builds, below-fold lazy images.
- Signed expiring object URL support (`/objects/employee/<secure_id>/profile/`), health/readiness endpoints, JSON structured logging, Sentry opt-in, custom 400/403/404/500, JS crash message.
- Legacy server camera stream and insecure CLI kiosk disabled. Use local browser getUserMedia. Server processes must never access a shared camera.

## Required limits to understand

1. **Unattended biometric attendance is intentionally blocked by default** (`REQUIRE_VERIFIED_LIVENESS=True`), because the supplied liveness class is not a validated anti-spoof/PAD implementation. A face match or simple blink/motion is not sufficient to defend against photo/video/deepfake/replay attacks. Integrate and independently test a suitable PAD solution before enabling this workflow; turning off the setting for a live public system is not recommended. The browser scanner currently returns `503 liveness_required` until integrated. HR can use authenticated audited manual attendance as a fallback.
2. No software configuration can guarantee **zero leaks**, **100% recognition**, uptime during outages, or perfect recognition across demographics/lighting. Perform demographic fairness tests, calibration, spoof/attack tests, and load tests with your own hardware and licensed face models.
3. InsightFace model packs can have **non-commercial licence restrictions**. Independently obtain appropriate licences before organisational/commercial use.
4. Images supplied earlier in `/media` must be relocated to `private_media/` and must **not** be served via a web server `alias /media/`. Review legacy `JoiningDocument`/salary exports and permissions before rollout.
5. **Every legacy numeric URL remains for existing templates**, which still use normal signed-in tenant permissions. New signed object links are available, but changing link appearance does not replace server authorization.
6. Browser GPS coordinates are client provided, not cryptographic proof of location. Use managed devices, device attestation and physical-presence checks where appropriate.
7. The existing app is large; migrations, all payroll/leave edge cases, permission matrix, HTML templates and end-to-end workflows require staging tests before production.

## First-time setup (Ubuntu 24.04 / Python 3.12)

```bash
sudo apt update
sudo apt install -y python3.12-venv build-essential cmake libgl1 libglib2.0-0 libpq-dev postgresql redis-server
cd /srv/face-attendance
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel
pip install -r requirements.txt
cp .env.example .env
chmod 600 .env
# Edit .env with your REAL secrets and infrastructure; never commit it.
python -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())"
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
# Put outputs into SECRET_KEY and FACE_ENCRYPTION_KEY respectively.
```

Create a PostgreSQL database and restricted app account. Place production DB credentials into `.env`. Configure an actual Redis instance/password, `ALLOWED_HOSTS`, `CSRF_TRUSTED_ORIGINS` (exact HTTPS origin), `DB_SSLMODE` (based on your local/remote Postgres setup), and a TLS reverse proxy. PostgreSQL migrations need owner rights for the attendance INSERT/UPDATE trigger.

**Before updating an existing database**: Take a verified encrypted DB backup AND a private media backup, freeze writes, inspect `audit_tenant_integrity`, and retain the `FACE_ENCRYPTION_KEY` in an external secret vault/backup. Do not rotate this key without re-encrypting old records. Read the migration source in `core/migrations/0034_biometric_encryption_and_indexes.py`.

```bash
# Stop old app writers. BACK UP DB before changing anything.
python manage.py check --deploy
python manage.py audit_tenant_integrity
python manage.py migrate --plan
python manage.py migrate
python manage.py audit_tenant_integrity
python manage.py collectstatic --noinput
python manage.py check --deploy
python manage.py test
```

The new encrypted `face_encoding` field converts legacy plaintext values during migration 0034. Do not roll back that migration without a reviewed decrypt/data migration. If an old deployment used a separate `MEDIA_ROOT`, securely move existing files and update storage paths, for example **only after checking both directory trees**:

```bash
mkdir -p private_media
rsync -a --chmod=D700,F600 media/ private_media/
chmod 700 private_media
```

This archive deliberately contains no `.env`, `.git`, `venv/`, production logs, biometric images, database files, or model weights. It is a source archive, not a database backup. New face images are WebP-encoded/stripped of EXIF and kept out of public static file serving.

## Inference capacity and availability

Face model files are provisioned **offline** into `$INSIGHTFACE_MODEL_ROOT/models/$INSIGHTFACE_MODEL/` before starting Gunicorn. Do not auto-download a model per request. Each Gunicorn worker loads its own model into memory at first inference. Begin with **two workers** if RAM permits, observe resident memory, and scale by adding replicas in front of PostgreSQL+Redis. Redis supports the cross-worker confirmation window; PostgreSQL serialises writes to the same employee. Test with 10/50/100 simultaneous clients and adjust per-worker memory, model inference slots and timeouts. Do not send full video streams to Django at 30 FPS; the browser compresses 1 frame per scan.

Review the supplied Nginx and systemd templates in `deploy/`; update paths, certificates and security policy with an experienced operator. Use a monitor on `GET /readyz/` and check alerting, backup restore procedures, secrets rotation, data retention/deletion, consent notices and audit logging. A 200 response is not proof of the AI model functioning or kiosk liveness.

## Security/quality verification matrix

| Test | Expected outcome |
|---|---|
| Cookie-based auth, CSRF and login throttling | unauthenticated/CSRF-invalid requests rejected; 429 after rate limit |
| Tenant-A user requests Tenant-B employee/payroll/signed link | 403 or 404; no tenant-B results |
| Tenant-A vs Tenant-B overlapping `employee_id` | scoped by internal employee PK + organisation |
| Concurrency: same employee 20 overlapping scans | no extra row during cooldown; no duplicate check-in |
| Concurrent independent organisations 10/50/100 clients | scan sessions remain isolated; service degrades with 503 rather than OOM |
| Unknown or ambiguous face, multiple people, spoof/replay | no attendance; PAD gate required to be deployed/tested |
| Oversized/corrupt image, GPS NaN, invalid session | 400, no stored image |
| Redis/Postgres outage | fail-closed or readiness 503; alert |
| Missing model files or too many parallel inferences | scanner returns 503 and remains responsive |
| Private-media URL without permission | 404; never direct file serving from Nginx |
| Encrypted DB dump inspection | stored `face_encoding` fields begin `enc:v1:` |
| No internet/static assets during deployment | main app stays up; see CSP and CDN note below |

**Current test limitation:** On 2026-10-18, Django `manage.py check`, migration checks, `pip check`, Python compilation, a production-mode `collectstatic` build, and all 21 Django tests passed in the local SQLite environment with Django 5.2.18. A browser smoke test covered login rendering, password visibility, invalid-password handling, redirects, health/readiness endpoints, and the favicon. Production-mode `check --deploy` reports only the HSTS subdomain and preload policy warnings. Real PostgreSQL/Redis, production credentials, end-to-end biometric scans, full CSP evaluation, concurrency under PostgreSQL, and realistic load testing **have not been executed here**. Validate these on staging before deployment. Templates still fetch some CSS/JS/fonts from CDNs; for strict offline availability and Subresource Integrity, vendor them locally and remove CSP CDN exceptions.

## Monitoring and privacy

- `GET /healthz/`: HTTP worker process responds (no DB check).
- `GET /readyz/`: PostgreSQL + Redis basic ping; monitor externally for uptime.
- Structured JSON logs to stdout for log collectors; no camera frame or embeddings intentionally logged by the new request middleware. Audit old view messages before retaining logs long-term.
- Set `SENTRY_DSN` optionally and make sure its data processing/privacy terms suit biometric/employee data; disable session replay and PII capturing.
- Existing attendance dashboards provide basic analytics. Implement retention/consent and approved delete/subject-access workflows appropriate to applicable laws.

## Common migration blockers

- `SECRET_KEY is required`: produce a cryptographically strong secret of 50+ chars.
- `FACE_ENCRYPTION_KEY required`: generate a 32-byte Fernet key once, secure it in a vault; **do not replace** on every restart.
- `Unencrypted biometric record`: migration 0034 did not complete or old scripts inserted plaintext; stop app and repair the record.
- `Face model unavailable`: provision a licensed model in the configured root and verify permissions.
- `503 liveness_required`: add a vetted PAD service; this is **intentional fail closed**.
- `OperationalError DB mismatch`: audit before migration; rectify wrongly assigned tenant records before enabling the new PostgreSQL attendance trigger.


### Reverse proxy caution

`TRUST_PROXY_SSL_HEADER=True` is safe ONLY if trusted Nginx **overwrites** `X-Forwarded-Proto` and the Django/Gunicorn port cannot be reached directly by untrusted clients. Otherwise disable it and have the TLS proxy and app deployment reviewed.

## October 2026 payroll/calendar/UI reliability upgrade (migration 0035)

This package also addresses the previously reported payslip and calendar issues:

- **Payroll generation** is explicit: creating a salary structure no longer creates a misleading automatic "processed" payroll. Regeneration is tenant-scoped and idempotent; it updates only unpaid records, never bulk-deletes payroll and never overwrites paid records. The organisation row is locked during generation on PostgreSQL. Finance users must run it with an active organisation membership.
- **Calculation policy (requires employer sign-off):** calendar-day proration of base salary/HRA/allowances; weekends and recognised holidays are paid rest days; approved paid leave is paid; half days earn 0.5; unpaid leave and absence reduce fixed pay. Regular overtime uses configured hourly rate, eligible weekend/holiday overtime uses 1.5x. Salary must be reviewed for statutory taxes, benefits, local labour law, effective-dated pay rises and organisation-specific policy before paying. OT pay and premium are saved in explicit snapshot columns.
- **Payslip:** PDF is always generated from the persisted payroll (not a possibly stale saved file); amounts reconcile to the recorded net and a legacy adjustment is disclosed if historical rows do not add up. Payment confirmation cannot silently alter `net_salary`; partial/overpayments need a separate adjustment/ledger process.
- **Calendar:** overnight check-in/out sessions are attributed to the check-in work date; open/unmatched check-ins are flagged `Pending checkout` instead of accruing automatically paid hours. Calendar and payroll share the same session attribution engine. Tenant-specific permission checks now also protect day-detail and calendar JSON endpoints.
- **Scanner bursts:** short bounded inference wait and adaptive browser backoff on 429/503; per-device and per-account rate limits instead of only one restrictive shared account rate. Redis-backed per-device/per-organisation confirmation remains isolated across workers. Actual throughput still depends on CPU/GPU, RAM, worker count, PAD solution and production load testing.
- **UI:** responsive design stylesheet (`static/css/ui-refresh.css`) across dashboards, cards, forms and salary tables; compact seven-day calendar grid for narrow phones; scanner page no longer blocks user zoom. Run `collectstatic` after deployment.
- **HR privacy:** Managers no longer automatically receive visibility of their reports' salaries in the payroll list; employees can see only their own. Finance and organisation Admin roles can see the organisation's payroll.

### Required upgrade steps for an existing database

1. Stop old web and worker processes; verify encrypted PostgreSQL and private-media backups.
2. Deploy the complete source tree and keep the **same** `FACE_ENCRYPTION_KEY`. Run `python manage.py migrate --plan`, then `python manage.py migrate` (adds **0035**).
3. Run `python manage.py collectstatic --noinput` (the new CSS is hashed by the manifest storage). Restart Gunicorn and verify Nginx static serving.
4. For each organisation/month, use `python manage.py audit_payroll_integrity --organisation 1 --year 2026 --month 10`. Historical **paid** payroll is intentionally unchanged. Migration 0035 initializes overtime snapshots to zero for old rows, so any historical paid salary with overtime needs finance-approved reconciliation and an audited adjustment workflow, not blind regeneration.
5. For the selected unpaid periods, regenerate payroll through Finance > Generate Payroll. **Spot-check 3 employee examples** (regular attendance, approved leave, night shift/overtime) against official payslips and finance rules before marking anything paid.
6. Verify actual PAD integration and run concurrent scans using distinct browser sessions/accounts/devices across different organisations. `REQUIRE_VERIFIED_LIVENESS=True` still prevents unattended biometric attendance **until a validated PAD provider is integrated**. Do not set it to false in production merely to make scans pass. For immediate operations use audited authorised manual attendance.
7. Run `python manage.py test core.tests.test_payroll_calendar_concurrency core.tests.test_security_upgrade` with Django test dependencies. For realistic 10/50/100-device capacity and deadlock tests, use staging PostgreSQL + Redis + model weights. SQLite unit tests do not prove these properties.

### What was verified in this environment

`manage.py check`, migration graph/dry-run consistency and all 21 tests passed with Django 5.2.18 in the local SQLite environment; an actual payslip PDF header was checked. The login browser smoke test had no JavaScript console errors. Production `collectstatic` succeeded. This does **not** verify regulatory payroll calculations for your specific organisation, real anti-spoofing, PostgreSQL row-lock contention, Redis outage recovery, smartphone/browser layouts beyond the login page, or production scalability.


## 28 October 2026: attendance, permissions, audit and payroll upgrade

Migration **0036_audit_history** adds organisation-scoped audit history at `/audit/`.
Organisation Admin users see only their organisation; platform superusers can inspect all organisations.
Audit events contain actor, time, request ID, entity, correction reason and before/after values.
They cover model-based employee, attendance, salary, payroll, work-rule, shift, leave, department,
location and resignation changes, plus login/logout and account changes through employee management.
Passwords, embeddings and camera images are excluded. QuerySet bulk updates bypass Django signals;
external scripts must use model saves or explicitly write audit events. PostgreSQL adds an UPDATE/DELETE
rejection trigger; use a restricted application database role. Database owners can still override database
protections, so forward audit events to separately controlled storage if tamper evidence is required.
History starts at installation and does not reconstruct earlier actions.

### Behaviour and permission changes

- Only an employee with the **Admin** role can correct their own attendance. HR can correct other
  employees in their organisation. Finance, Manager and Employee cannot manually correct attendance.
  Face-based check-in/out remains available to authenticated staff when the model and verified PAD are configured.
- All corrections require a reason, validate timestamps/types, use local time, and lock the employee row.
  Historical attendance can be corrected, including dates in paid periods. Paid payroll remains an
  immutable snapshot; processed unpaid payroll is returned to pending for recalculation. Neighbouring
  dates that may affect overnight sessions are included when identifying affected payroll periods.
- Calendar, JSON details and dashboard working hours share completed-session calculations. The calendar
  distinguishes present, half day, absent, insufficient hours, pending checkout, inconsistent logs,
  paid/unpaid leave, rest day, holiday, future dates and dates outside employment. Overnight details show
  the actual date of both events. Duplicate and orphan events are visible and block payroll until reconciled.
- Payroll policy is preserved as requested: calendar-day proration; paid rest days/holidays;
  90% of the assigned full-day hours qualifies as present; configured half-day threshold; approved paid
  leave paid and unpaid leave unpaid; regular overtime at the applicable shift rate and eligible
  holiday/weekend overtime at 1.5x. Deductions are capped at earned gross so net reconciles exactly.
  Excess deductions are not carried forward; a separate receivables ledger would be needed for that policy.
- Generation locks employees before reading attendance. Approved unpaid payroll becomes pending when
  source attendance, salary, leave or shift information changes. **Current/future months are provisional**
  and cannot be marked paid. Regenerate after month close and source reconciliation.
- Salary changes that start partway through an existing employee's month are refused for generation:
  the current single salary structure cannot reconstruct an earlier pay rate. Effective-dated salary
  history is a remaining requirement for that scenario; do not silently apply the new rate to earlier days.
- Paid payroll cannot be regenerated, deleted or paid twice. Manual payroll edits are disabled because
  counts and amounts are source-derived. Correct the underlying inputs and regenerate unpaid payroll.
  Statutory tax/PF/ESI and legal compliance are not newly implemented by this upgrade.
- Finance/Admin can access salary setup and payroll processing. Employees can view their own payslips.
  HR/Manager cannot inspect another employee's salary through salary setup or dashboard selection.
- Django admin attendance and payroll mutations are disabled: use the audited application workflow.
  HR cannot reset privileged accounts or change their own official employee profile. Only Admin assigns roles.

### Scanner deployment

`FACE_SCAN_INFERENCE_SLOTS` (1–8, default 1) controls independent model instances per worker.
Unlike a shared inference mutex, two slots can process two requests concurrently. Each instance consumes
model memory: select worker/slot counts only after measuring CPU/GPU and RAM on staging.
Only detection and recognition modules are loaded. `FACE_EXECUTION_PROVIDERS` defaults to
`CPUExecutionProvider`; GPU use requires the corresponding ONNX Runtime build and drivers.
Gallery versions are shared via Redis and invalidate on employee saves after transaction commit.
Gallery queries no longer hold a cross-organisation cache lock. Votes are isolated by organisation/session,
reset after failed frames or identity changes, and require consecutive matching frames.
Pool exhaustion and shared gallery-state outages return retryable busy responses.

The original project-local face-model directory was empty. The local setup now points to the existing
`C:/Users/avane/.insightface` model cache, and real model loading and blank-frame inference have been tested.
The local `.env` explicitly enables **supervised face-only mode** with `REQUIRE_VERIFIED_LIVENESS=False`
because no anti-spoofing provider is available. This does not provide photo/video spoof resistance.
The scanner displays that limitation in normal and fullscreen modes. Deployment defaults in
`.env.example` still require verified PAD. For that mode, configure `FACE_LIVENESS_VERIFIER=your_package.pad.verify` with a
server-side callable `verify(*, frame, request, organisation) -> bool`. It must verify freshness,
frame/session/organisation binding and replay resistance with a bounded provider timeout.
Only the literal boolean `True` authorizes a frame. Provider failures fail closed. This interface is not
itself a liveness detector; do not substitute a client-provided flag or disable the gate for production.

### Upgrade and verification

Back up the database and private media, then deploy the source and run:

```bash
python manage.py migrate --plan
python manage.py migrate
python manage.py collectstatic --noinput
python manage.py check
python manage.py audit_tenant_integrity
python manage.py test
```

Restart all workers after deployment. No existing payroll is automatically regenerated or marked paid.
Audit records require migration 0036 before the application accepts writes or user logins.

Local validation on 28 October 2026 uses SQLite and mock face models. Regression tests cover permission
boundaries, audit atomicity/isolation, invalid corrections, completed and incomplete sessions, overnight
attribution, rotation rates, full-month salary reconciliation, source invalidation, protected payment,
independent inference slots, failure backpressure, gallery invalidation and PAD fail-closed behaviour.
Browser checks use a separate synthetic SQLite database, including calendar rendering, overnight modal,
correction save, audit page, desktop/mobile layouts and browser console errors.
PostgreSQL/Redis concurrency, trigger execution, real inference/PAD, sustained device load, recognition
accuracy and model licensing still require deployment-specific validation. Unit tests do not certify these.

Employee removal now deactivates access and retains payroll/attendance history. Hard deletion of employees
with attendance or payroll is rejected, including account cascades. Reactivation requires a POST confirmation.

Local verification: 54 templates compiled; JavaScript syntax checked; 390px calendar has no horizontal
overflow; calendar correction and audit pages showed no browser console errors. Local database migration
0036 was applied after a SQLite backup at `.backups/dev-before-audit-0036.sqlite3`.

`check --deploy` on this local DEBUG/SQLite configuration reports five expected warnings for DEBUG,
HTTPS redirect, HSTS and secure cookies. It is a development environment, not a deployed production
instance. Use the production PostgreSQL/Redis/HTTPS configuration and rerun deployment checks on staging.


### Repairing enrolment after model setup

A stored photo with `FACE_PENDING` is not a usable face template. The scanner now reports an empty
organisation gallery distinctly from an unknown face. To repair existing missing/pending templates
without replacing valid enrolments, preview then apply:

```bash
python manage.py repair_face_enrollments --organisation 1
python manage.py repair_face_enrollments --organisation 1 --apply
```

Use `--employee <internal employee primary key>` to target one employee. Only active employees in the
specified organisation are considered. A saved photo containing one clear face is required. Repairs
are encrypted and audited without storing embeddings in the audit event. This command never marks
attendance. After a repair, refresh the scanner; its gallery reloads within 30 seconds across local
workers, or immediately through shared Redis invalidation in production.


## Django admin access model

- Global superusers have full Django admin access across organisations, including create, edit and delete operations on registered business models. Organisation records are visible only to global superusers.
- An active staff user whose Employee profile has the `Admin` role can access the Django admin with tenant-scoped access. The admin index exposes only organisation-owned models; querysets and related-object selectors are scoped to that organisation. The organisation field is omitted from tenant-admin forms and tenant ownership is assigned server-side.
- Django authentication users/groups/permissions and the audit-event model are not exposed to organisation admins. Audit history in the application is restricted to global superusers.
- Attendance and payroll records remain read-only for organisation admins in Django admin; global superusers can manage them. Use the application workflows for tenant attendance corrections and payroll processing. Audit events remain append-only and cannot be edited or deleted, including by superusers.
- Users must be marked `is_staff=True` to sign in to Django admin. Do not grant organisation admins `is_superuser`, Django auth-management permissions, or access through shared staff accounts.

After deployment, run migrations and collect static assets as usual. The custom Django admin stylesheet is served from `static/admin/css/production_admin.css`.
