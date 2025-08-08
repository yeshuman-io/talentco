from django.db import migrations, models
import uuid
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('organisations', '0001_initial'),
        ('opportunities', '0001_initial'),
        ('profiles', '0001_initial'),
        ('evaluations', '0001_initial'),
        ('auth', '0012_alter_user_first_name_max_length'),
    ]

    operations = [
        migrations.CreateModel(
            name='StageTemplate',
            fields=[
                ('id', models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False, serialize=False)),
                ('name', models.CharField(max_length=100)),
                ('slug', models.SlugField(unique=True)),
                ('order', models.PositiveIntegerField(default=0)),
                ('is_terminal', models.BooleanField(default=False)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
            ],
            options={'ordering': ['order', 'name']},
        ),
        migrations.CreateModel(
            name='Application',
            fields=[
                ('id', models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False, serialize=False)),
                ('status', models.CharField(max_length=20, choices=[('applied', 'Applied'), ('in_review', 'In Review'), ('interview', 'Interview'), ('offer', 'Offer'), ('hired', 'Hired'), ('rejected', 'Rejected'), ('withdrawn', 'Withdrawn')], default='applied', db_index=True)),
                ('source', models.CharField(max_length=20, choices=[('direct', 'Direct'), ('referral', 'Referral'), ('internal', 'Internal'), ('import', 'Import'), ('agency', 'Agency')], default='direct', db_index=True)),
                ('evaluation_snapshot', models.JSONField(blank=True, default=dict)),
                ('applied_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('evaluation', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='applications', to='evaluations.evaluation')),
                ('opportunity', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='applications', to='opportunities.opportunity')),
                ('organisation', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='applications', to='organisations.organisation')),
                ('profile', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='applications', to='profiles.profile')),
            ],
        ),
        migrations.CreateModel(
            name='ApplicationStageInstance',
            fields=[
                ('id', models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False, serialize=False)),
                ('entered_at', models.DateTimeField(auto_now_add=True)),
                ('exited_at', models.DateTimeField(blank=True, null=True)),
                ('application', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='stages', to='applications.application')),
                ('entered_by', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='entered_application_stages', to='auth.user')),
                ('stage_template', models.ForeignKey(on_delete=django.db.models.deletion.PROTECT, related_name='instances', to='applications.stagetemplate')),
            ],
        ),
        migrations.AddField(
            model_name='application',
            name='current_stage_instance',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='current_for_applications', to='applications.applicationstageinstance'),
        ),
        migrations.CreateModel(
            name='ApplicationEvent',
            fields=[
                ('id', models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False, serialize=False)),
                ('event_type', models.CharField(max_length=30, choices=[('applied', 'Applied'), ('stage_changed', 'Stage Changed'), ('decision_made', 'Decision Made'), ('withdrawn', 'Withdrawn'), ('note_added', 'Note Added')], db_index=True)),
                ('metadata', models.JSONField(blank=True, default=dict)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('actor', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='application_events', to='auth.user')),
                ('application', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='events', to='applications.application')),
            ],
            options={'ordering': ['-created_at']},
        ),
        migrations.CreateModel(
            name='OpportunityQuestion',
            fields=[
                ('id', models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False, serialize=False)),
                ('question_text', models.TextField()),
                ('question_type', models.CharField(max_length=20, choices=[('text', 'Text'), ('boolean', 'Boolean'), ('single_choice', 'Single Choice'), ('multi_choice', 'Multi Choice'), ('number', 'Number')])),
                ('is_required', models.BooleanField(default=False)),
                ('order', models.PositiveIntegerField(default=0)),
                ('config', models.JSONField(blank=True, default=dict)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('opportunity', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='questions', to='opportunities.opportunity')),
            ],
            options={'ordering': ['opportunity', 'order', 'id']},
        ),
        migrations.CreateModel(
            name='ApplicationAnswer',
            fields=[
                ('id', models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False, serialize=False)),
                ('answer_text', models.TextField(blank=True)),
                ('answer_options', models.JSONField(blank=True, default=list)),
                ('is_disqualifying', models.BooleanField(default=False)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('application', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='answers', to='applications.application')),
                ('question', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='answers', to='applications.opportunityquestion')),
            ],
        ),
        migrations.CreateModel(
            name='Interview',
            fields=[
                ('id', models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False, serialize=False)),
                ('round_name', models.CharField(default='Interview', max_length=100)),
                ('scheduled_start', models.DateTimeField()),
                ('scheduled_end', models.DateTimeField()),
                ('location_type', models.CharField(choices=[('virtual', 'Virtual'), ('onsite', 'Onsite')], default='virtual', max_length=10)),
                ('location_details', models.CharField(blank=True, max_length=255)),
                ('outcome', models.CharField(choices=[('pending', 'Pending'), ('pass', 'Pass'), ('fail', 'Fail')], default='pending', max_length=10)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('application', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='interviews', to='applications.application')),
            ],
        ),
        migrations.AddConstraint(
            model_name='application',
            constraint=models.UniqueConstraint(fields=('profile', 'opportunity'), name='uniq_profile_opportunity_application'),
        ),
        migrations.AddIndex(
            model_name='application',
            index=models.Index(fields=['organisation', 'status'], name='applications_org_status_idx'),
        ),
        migrations.AddIndex(
            model_name='application',
            index=models.Index(fields=['opportunity', 'status'], name='applications_opp_status_idx'),
        ),
        migrations.AddConstraint(
            model_name='applicationanswer',
            constraint=models.UniqueConstraint(fields=('application', 'question'), name='uniq_application_question_answer'),
        ),
    ]
