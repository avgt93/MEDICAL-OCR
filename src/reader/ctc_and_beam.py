from dataclasses import dataclass, field
from itertools import groupby
from typing import List

import numpy as np

from src.reader.prefix_tree import PrefixTree


@dataclass
class Beam:
    text: str
    prob_blank: float
    prod_non_blank: float

    @property
    def prob_total(self) -> float:
        return self.prob_blank + self.prod_non_blank


def ctc_single_word_beam_search(
    predictions: np.ndarray, chars: List[str], beam_width: int, prefix_tree: PrefixTree
):
    res = []
    for batch_idx in range(predictions.shape[1]):
        num_timesteps = predictions.shape[0]
        prev = [Beam("", 1, 0)]

        for time_idx in range(num_timesteps):
            curr = []

            best_beams = sorted(prev, key=lambda x: x.prob_total, reverse=True)[
                :beam_width
            ]
            for beam in best_beams:
                pr_non_blank = 0
                if beam.text != "":
                    label_idx = chars.index(beam.text[-1]) + 1
                    pr_non_blank = (
                        beam.prod_non_blank
                        * predictions[time_idx, batch_idx, label_idx]
                    )

                pr_blank = beam.prob_total * predictions[time_idx, batch_idx, 0]

                curr.append(Beam(beam.text, pr_blank, pr_non_blank))

                next_chars = prefix_tree.get_next_chars(beam.text)
                for c in next_chars:
                    label_idx = chars.index(c) + 1
                    if beam.text != "" and beam.text[-1] == c:
                        pr_non_blank = (
                            predictions[time_idx, batch_idx, label_idx]
                            * beam.prob_blank
                        )
                    else:
                        pr_non_blank = (
                            predictions[time_idx, batch_idx, label_idx]
                            * beam.prob_total
                        )

                    curr.append(Beam(beam.text + c, 0, pr_non_blank))

            prev = curr

        prev = [
            beam for beam in prev if prefix_tree.is_word(beam.text)
        ]  # only keep words
        best_beams = sorted(prev, key=lambda x: x.prob_total, reverse=True)
        res.append(best_beams[0].text if best_beams else "")

    return res


def ctc_best_path(predictions: np.ndarray, chars: List[str]) -> List[str]:
    res = []
    for b in range(predictions.shape[1]):
        best_path = np.argmax(predictions[:, b], axis=1)

        best_path_decoded = [chars[c - 1] for c, _ in groupby(best_path) if c != 0]

        text = "".join(best_path_decoded)
        res.append(text)

    return res
