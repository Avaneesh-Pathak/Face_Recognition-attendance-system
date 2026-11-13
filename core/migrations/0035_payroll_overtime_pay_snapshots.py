from decimal import Decimal
from django.db import migrations, models

class Migration(migrations.Migration):
    dependencies = [('core', '0034_biometric_encryption_and_indexes')]
    operations = [
        migrations.AddField(model_name='payroll', name='regular_overtime_pay',
            field=models.DecimalField(max_digits=10, decimal_places=2, default=Decimal('0'), help_text="Snapshot: paid regular overtime amount")),
        migrations.AddField(model_name='payroll', name='holiday_overtime_pay',
            field=models.DecimalField(max_digits=10, decimal_places=2, default=Decimal('0'), help_text="Snapshot: holiday/weekend overtime premium amount")),
    ]
