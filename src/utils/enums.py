from enum import Enum

class ModelType(Enum):
    RANDOMFOREST = "randomforest"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"

    @classmethod
    def validation(cls, value: str):
        value = value.lower()
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(f"❌ Invalid model_type: {value}. Must be one of : {[m.value for m in cls]}")