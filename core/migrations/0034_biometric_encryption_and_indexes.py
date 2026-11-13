from django.db import migrations, models
import core.security.biometrics
import os


def migrate_legacy_templates(apps, schema_editor):
    # One-time conversion. Back up DB and retain the encryption key beforehand.
    # Query through the new field while the migration alone permits legacy reads.
    Employee = apps.get_model('core', 'Employee')
    previous = os.environ.get('ALLOW_LEGACY_FACE_TEMPLATES')
    os.environ['ALLOW_LEGACY_FACE_TEMPLATES'] = '1'
    try:
        for employee in Employee.objects.exclude(face_encoding__isnull=True).exclude(face_encoding='').iterator(chunk_size=200):
            if employee.face_encoding:
                employee.save(update_fields=['face_encoding'])
    finally:
        if previous is None:
            os.environ.pop('ALLOW_LEGACY_FACE_TEMPLATES', None)
        else:
            os.environ['ALLOW_LEGACY_FACE_TEMPLATES'] = previous


def install_tenant_guard(apps, schema_editor):
    if schema_editor.connection.vendor != 'postgresql':
        return
    with schema_editor.connection.cursor() as cursor:
        cursor.execute('''
        CREATE OR REPLACE FUNCTION core_guard_attendance_org() RETURNS trigger AS $$
        BEGIN
          IF NEW.organisation_id IS NULL OR NOT EXISTS (
             SELECT 1 FROM core_employee e
             WHERE e.id=NEW.employee_id AND e.organisation_id=NEW.organisation_id
          ) THEN
             RAISE EXCEPTION 'cross-organisation attendance is forbidden';
          END IF;
          RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        DROP TRIGGER IF EXISTS attendance_tenant_guard ON core_attendance;
        CREATE TRIGGER attendance_tenant_guard BEFORE INSERT OR UPDATE ON core_attendance
          FOR EACH ROW EXECUTE FUNCTION core_guard_attendance_org();
        ''')

class Migration(migrations.Migration):
    dependencies = [('core', '0033_organisation_address_organisation_alternate_phone_and_more')]
    operations = [
        migrations.AlterField(model_name='employee', name='face_encoding',
            field=core.security.biometrics.EncryptedFaceEncodingField(blank=True, null=True)),
        migrations.RunPython(migrate_legacy_templates, migrations.RunPython.noop),
        migrations.AddIndex('employee', models.Index(fields=['organisation', 'is_active', 'employment_status'], name='emp_org_active_idx')),
        migrations.AddIndex('attendance', models.Index(fields=['organisation', '-timestamp'], name='att_org_ts_idx')),
        migrations.AddIndex('attendance', models.Index(fields=['organisation','employee','-timestamp'], name='att_org_emp_ts_idx')),
        migrations.AddIndex('payroll', models.Index(fields=['organisation','-month','status'], name='payroll_org_month_idx')),
        migrations.AddIndex('leaveapplication', models.Index(fields=['organisation','status','-applied_on'], name='leave_org_status_idx')),
        migrations.AddIndex('attendancematchlog', models.Index(fields=['organisation','-created_at'], name='matchlog_org_ts_idx')),
        migrations.RunPython(install_tenant_guard, migrations.RunPython.noop),
    ]
