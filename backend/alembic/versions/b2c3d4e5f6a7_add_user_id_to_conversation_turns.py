"""add_user_id_to_conversation_turns

Revision ID: b2c3d4e5f6a7
Revises: a1b2c3d4e5f6
Create Date: 2026-10-04 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b2c3d4e5f6a7'
down_revision: Union[str, Sequence[str], None] = 'a1b2c3d4e5f6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema to add user_id to conversation_turns."""
    op.add_column('conversation_turns', sa.Column('user_id', sa.UUID(), nullable=True))
    op.create_index(op.f('ix_conversation_turns_user_id'), 'conversation_turns', ['user_id'], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f('ix_conversation_turns_user_id'), table_name='conversation_turns')
    op.drop_column('conversation_turns', 'user_id')
