from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

from src.reader import read
from src.reader.ctc_and_beam import PrefixTree
from src.detector import detect, sort_multiline, AABB


@dataclass
class WordReadout:
    text: str
    aabb: AABB


@dataclass
class Detector:
    scale: float = 1.0
    margin: int = 0


@dataclass
class LineClustering:
    min_words_per_line: int = 1
    max_dist: float = 0.7


@dataclass
class Reader:
    decoder: str = "best_path"
    prefix_tree: Optional[PrefixTree] = None


def page_reader(
    img: np.ndarray,
    detector_config: Detector = Detector(),
    line_clustering_config=LineClustering(),
    reader_config=Reader(),
) -> List[List[WordReadout]]:
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    detections = detect(img, detector_config.scale, detector_config.margin)

    lines = sort_multiline(
        detections, min_words_per_line=line_clustering_config.min_words_per_line
    )
    read_lines = []
    for line in lines:
        read_lines.append([])
        for word in line:
            text = read(word.img, reader_config.decoder, reader_config.prefix_tree)
            read_lines[-1].append(WordReadout(text, word.aabb))

    return read_lines
