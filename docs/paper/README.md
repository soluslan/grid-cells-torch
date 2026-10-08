# The paper

This project reimplements:

> Andrea Banino, Caswell Barry, Benigno Uria, Charles Blundell, Timothy Lillicrap, Piotr Mirowski,
> Alexander Pritzel, Martin J. Chadwick, Thomas Degris, Joseph Modayil, Greg Wayne, Hubert Soyer,
> Fabio Viola, Brian Zhang, Ross Goroshin, Neil Rabinowitz, Razvan Pascanu, Charlie Beattie,
> Stig Petersen, Amir Sadik, Stephen Gaffney, Helen King, Koray Kavukcuoglu, Demis Hassabis,
> Raia Hadsell & Dharshan Kumaran.
> **Vector-based navigation using grid-like representations in artificial agents.**
> *Nature* 557, 429–433 (2018). https://doi.org/10.1038/s41586-018-0102-6

We are not the authors. The paper, its Supplementary Information, and its figures are the
work of the authors above and are published by Springer Nature; the original supervised-learning
code is [google-deepmind/grid-cells](https://github.com/google-deepmind/grid-cells)
(Apache-2.0). Everything in this folder that is committed to git is our own secondary material
(notes and summaries with page citations), not a copy of the paper.

```bibtex
@article{banino2018vector,
  title   = {Vector-based navigation using grid-like representations in artificial agents},
  author  = {Banino, Andrea and Barry, Caswell and Uria, Benigno and Blundell, Charles and
             Lillicrap, Timothy and Mirowski, Piotr and Pritzel, Alexander and Chadwick, Martin J.
             and Degris, Thomas and Modayil, Joseph and Wayne, Greg and Soyer, Hubert and
             Viola, Fabio and Zhang, Brian and Goroshin, Ross and Rabinowitz, Neil and
             Pascanu, Razvan and Beattie, Charlie and Petersen, Stig and Sadik, Amir and
             Gaffney, Stephen and King, Helen and Kavukcuoglu, Koray and Hassabis, Demis and
             Hadsell, Raia and Kumaran, Dharshan},
  journal = {Nature},
  volume  = {557},
  number  = {7705},
  pages   = {429--433},
  year    = {2018},
  doi     = {10.1038/s41586-018-0102-6}
}
```

## What is in this folder

| File | Committed | Contents |
|---|---|---|
| `spec-supervised.md` | yes | Supervised path-integration network: architecture, loss, Table 1, reported results |
| `spec-rl-agent.md` | yes | RL agent: environment, architecture, training, Table 2, reported results |
| `figures.md` | yes | Per-figure facts that only the image shows, and when to reopen each image |
| `deviations.md` | yes | Every place this code knowingly differs from the paper or the released code |
| `banino2018-main.pdf`, `banino2018-supplementary.pdf` | no (gitignored) | Local copies of the paper |
| `figures/*.png` | no (gitignored) | Figures cropped from the PDFs (`fig1`–`fig4`, `ed-fig1`–`ed-fig10`) |
| `text/main.txt`, `text/supp.txt` | no (gitignored) | `pypdf` text extraction with `===== PAGE N =====` markers, for `grep` |

Page citations in the notes use **PDF page numbers** (the `PAGE N` markers in `text/*.txt`),
written `main p.N` / `supp p.N`. The supplementary PDF's printed footer number is one less
than its PDF page number.

To regenerate the text extraction after placing the PDFs here:

```bash
python - <<'EOF'
from pypdf import PdfReader
for src, dst in [("banino2018-main.pdf", "main.txt"), ("banino2018-supplementary.pdf", "supp.txt")]:
    with open(f"docs/paper/text/{dst}", "w") as f:
        for i, page in enumerate(PdfReader(f"docs/paper/{src}").pages, 1):
            f.write(f"\n===== PAGE {i} =====\n" + page.extract_text())
EOF
```

Text extraction mangles some symbols: `10^-5` comes out as `10 5`, Greek letters and vector
arrows are often dropped or garbled, and equations are scattered across lines. When a number or
formula matters, check it against the PDF page itself.
