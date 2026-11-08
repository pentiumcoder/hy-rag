# Roadmap

Things I'd like to do next, roughly in order:

- Add more near-miss questions to the test set. There are only 6 distinct ones, so the
  numbers for declining rest on very few cases.
- Get the test questions written by someone who didn't write the sample documents. Right
  now both come from me, which makes the test easier than real questions would be.
- Label a few dozen answers by hand to check the grading model, which is often wrong about
  what counts as incorrect.
- Parse refusals that the model splits over two lines (see the TODO in
  `src/hyrag/generation/claims.py`).
- Cut the time per answer. Most of it goes to swapping the answering and checking models in
  and out of an 8 GB GPU; a smaller checking model would help.
- Let the system prompt name the user's own organisation instead of the sample company.
