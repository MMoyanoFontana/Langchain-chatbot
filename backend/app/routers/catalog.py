from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from app.db import get_db
from app.models import Provider, ProviderModel
from app.schemas import ProviderModelRead

router = APIRouter(prefix="/catalog", tags=["catalog"])


@router.get("/models", response_model=list[ProviderModelRead])
def list_active_models(db: Session = Depends(get_db)) -> list[ProviderModel]:
    models = db.scalars(
        select(ProviderModel)
        .join(Provider, Provider.id == ProviderModel.provider_id)
        .where(
            Provider.is_active.is_(True),
            ProviderModel.is_active.is_(True),
        )
        .options(joinedload(ProviderModel.provider))
        .order_by(Provider.display_name.asc(), ProviderModel.display_name.asc())
    ).all()
    return list(models)
