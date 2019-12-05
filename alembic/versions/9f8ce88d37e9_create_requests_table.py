"""create requests table

Revision ID: 9f8ce88d37e9
Revises: 
Create Date: 2019-12-05 16:41:19.128010

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import func

# revision identifiers, used by Alembic.
revision = '9f8ce88d37e9'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'requests',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('created_at', sa.DateTime(timezone=True)),
        sa.Column('prediction_duration', sa.String),
        sa.Column('edition_id', sa.Integer),
        sa.Column('apex_node_predictor_probabilities', sa.Text),
        sa.Column('branch_predictor_probabilities', sa.Text),
        sa.Column('predictions', sa.Text),
        sa.Column('api_version', sa.String),
    )


def downgrade():
    op.drop_table('requests')
