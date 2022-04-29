# OA-Mine: Open-World Attribute Mining for E-Commerce Products with Weak Supervision

---

<h4 align="center">
    <p>
        <a href="https://dl.acm.org/doi/10.1145/3485447.3512035">Paper</a> |
        <a href="./data">Data</a> |
        <a href="./slides.pdf">Slides</a> |
        <a href="https://youtu.be/vrDPV8EMLnA">Video</a>
    <p>
</h4>

---

This repo contains code for the paper: [OA-Mine: Open-World Attribute Mining for E-Commerce Products with Weak Supervision](https://dl.acm.org/doi/10.1145/3485447.3512035).

## Introduction
* This project works on *open-world product attribute mining* with weak supervision.
* Unlike previous work that requires attributes of interest to be specified (e.g., "I want to extract brands and colors"), we aim to find *both* new attributes (e.g., "what are possible attributes for TV products") and values (e.g., "4K UHD") from product titles on e-commerce cites.
* Our framework has two steps: [attribute value candidate generation](./candidate_generation) and [attribute value grouping](./value_grouping).
* Please check [here](./data) for dataset release.

## Citation
If you find our code or data useful, please cite:
```bibtex
@inproceedings{zhang2022oamine,
author = {Zhang, Xinyang and Zhang, Chenwei and Li, Xian and Dong, Xin Luna and Shang, Jingbo and Faloutsos, Christos and Han, Jiawei},
title = {OA-Mine: Open-World Attribute Mining for E-Commerce Products with Weak Supervision},
year = {2022},
isbn = {9781450390965},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3485447.3512035},
doi = {10.1145/3485447.3512035},
booktitle = {Proceedings of the ACM Web Conference 2022},
pages = {3153–3161},
numpages = {9},
keywords = {weak supervision., Open-world product attribute mining},
location = {Virtual Event, Lyon, France},
series = {WWW '22}
}
```