import re
import cv2
import gradio as gr
from typing import List
from src import (
    page_reader,
    Detector,
    LineClustering,
    Reader,
    PrefixTree,
)
import numpy as np
from typing import Tuple


def process_gradio_input(
    result_image: np.ndarray,
    clustering_scale: float,
    clutering_margin: int,
    use_word_beam: bool,
    min_words_per_line: int,
    text_size: float,
) -> Tuple[str, np.ndarray]:
    with open("data/medical_alpha.txt") as f:
        medical_word_list = [w.strip().upper() for w in f.readlines()]
    prefix_tree = PrefixTree(medical_word_list)

    with open("data/words_alpha.txt") as f:
        word_list = [w.strip().upper() for w in f.readlines()]
    prefix_tree = PrefixTree(word_list)

    read_lines = page_reader(
        result_image,
        detector_config=Detector(scale=clustering_scale, margin=clutering_margin),
        line_clustering_config=LineClustering(min_words_per_line=min_words_per_line),
        reader_config=Reader(
            decoder="word_beam_search" if use_word_beam else "best_path",
            prefix_tree=prefix_tree,
        ),
    )
    result_words = ""
    for read_line in read_lines:
        result_words += " ".join(read_word.text for read_word in read_line) + "\n"

    def filter_medical_words(result_words):
        result_words = result_words.lower().split()

        with open("data/words_alpha.txt", "r") as file:
            medical_word_list = set(file.read().split())

        filtered_words = [words for words in result_words if words in medical_word_list]

        return filtered_words

    final_word_list = " ".join(filter_medical_words(result_words)).upper()

    for i, read_line in enumerate(read_lines):
        for read_word in read_line:
            aabb = read_word.aabb
            cv2.rectangle(
                result_image,
                (aabb.xmin, aabb.ymin),
                (aabb.xmin + aabb.width, aabb.ymin + aabb.height),
                (255, 0, 0),
                2,
            )
            cv2.putText(
                result_image,
                read_word.text,
                (aabb.xmin, aabb.ymin + aabb.height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                text_size,
                color=(255, 0, 0),
            )

    return result_words, result_image
