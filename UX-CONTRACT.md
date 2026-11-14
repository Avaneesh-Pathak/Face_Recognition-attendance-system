# FaceAttend UX Contract

This contract covers the organisation chart, notifications, and audit history.

## Canonical UI Map

| Capability | Canonical owner | Source of truth | Allowed variants | Verification |
|---|---|---|---|---|
| Select/Listbox | Native `<select>` for organisation and simple filters | This contract and `DESIGN.md` | Platform-owned popup | Keyboard selection and native popup |
| Date | Native date input for date filters | This contract and `DESIGN.md` | Platform-owned calendar | Keyboard selection and active browser locale |
| Scrollbar | Global rules in `templates/base.html` | `DESIGN.md` | `scrollbar-gutter` for the chart viewport | Django check and browser inspection |
| Notifications | Header in `templates/base.html`; data in `core/context_processors.py` | This contract | Header preview and paginated activity list | Recipient scoping and read flow |
| Organisation chart | `core/org_chart.py` and `templates/ems/org_chart.html` | This contract plus `core/middleware.py` tenant membership | Employee tenant view; superuser organisation selector | Tenant query scope and superuser selection |
| Audit timeline | `core/attendance_views.py` and `templates/audit_log.html` | `core/models.py` append-only audit model | Platform superuser view | Superuser boundary and immutable event display |

Business and security sources: `core/middleware.py` establishes organisation membership, `core/tenant_utils.py` scopes tenant managers, and `core/models.py` defines employee, notification, and immutable audit records.

## Organisation chart

- An authenticated employee sees active employees and reporting lines from their server-established organisation membership. Client-supplied `organisation_id` values never change a tenant user's scope.
- A system superuser can select any active organisation. An unselected chart displays a selection prompt; an invalid or inactive organisation returns no employee data.
- Both APIs return only records from the selected organisation. Active employees with no active manager appear at the top level.
- Cards show the employee name, employee code, position, department, and direct-report count. Branches can be collapsed with a keyboard-accessible button. The chart supports zoom and two-axis scrolling on desktop and mobile.

## Notifications

- `core.context_processors.notification_context` is the shared owner for the header bell. It shows the current user's unread count and five most recent notifications. The count caps visually at `99+` and the accessible name retains the exact count.
- Opening a notification marks it read and redirects only to a safe same-site link. The notification list also provides explicit read actions; read mutations use POST.
- Leave submission notifies the assigned approver. Each subsequent workflow stage notifies its next approver. Final approval or rejection notifies the applicant.
- A notification can only be created for an active employee account in the same organisation as the notification record.

## Audit history

- Audit history is restricted to platform superusers and remains append-only.
- Each event shows its organisation, actor, employee subject when resolvable, employee code, object, action, and exact event time. Deleted employee details use the limited name/code values stored in the event snapshot.
- Audit snapshots do not add contact details, face data, images, or documents.

## Canonical owners

- Header bell and count: `templates/base.html` and `core/context_processors.py`.
- Notification creation and recipient scope: `core/notifications.py`.
- Organisation chart scope and tree response: `core/org_chart.py`.
- Audit event enrichment: `core/attendance_views.py` and `core/security/audit.py`.
