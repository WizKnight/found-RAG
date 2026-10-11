from functools import lru_cache

from app.core.config import get_settings
from app.verification.lexical_verifier import LexicalClaimVerifier
from app.verification.nli_verifier import NLIClaimVerifier
from app.verification.pipeline import VerificationPipeline


@lru_cache
def get_verification_pipeline() -> VerificationPipeline:
    settings = get_settings()

    if settings.verification_method == "lexical":
        verifier = LexicalClaimVerifier()

    elif settings.verification_method == "nli":
        verifier = NLIClaimVerifier(
            model_name=settings.nli_model,
            device=settings.nli_device,
        )

    else:
        raise ValueError(
            "Unsupported verification method: "
            f"{settings.verification_method}"
        )

    return VerificationPipeline(
        verifier=verifier,
    )