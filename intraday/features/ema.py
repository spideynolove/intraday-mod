from typing import Sequence, Union, Literal
from intraday.frame import Frame
from intraday.feature import StatefulEMA


class EMA(StatefulEMA):
    def __init__(
        self,
        period: int,
        source: Union[str, Sequence[str]],
        write_to: Literal["frame", "state", "both"] = "state",
    ):
        super().__init__(period, source, "ema", write_to)

    def extract_value(self, frames: Sequence[Frame], source_name: str):
        return getattr(frames[-1], source_name)
