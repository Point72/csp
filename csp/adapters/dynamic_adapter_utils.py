from pydantic import BaseModel, NonNegativeInt


class AdapterInfo(BaseModel):
    caller_id: NonNegativeInt
    is_subscribe: bool
