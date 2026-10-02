# MHCflurry 2.3.13

This documentation-only release removes filler from the user guides and fixes
what review found in that change. 2.3.12 was merged but not published; its
changes are included here. Library behavior, model weights and predictions are
unchanged; default weights remain 2.3.0.

- Removed sentences that only reassured readers ("most users…"), described the
  page itself, or repeated a nearby sentence, across the landing page,
  tutorials, guides, references and maintainer pages. The landing page now lists
  the Advanced topics.
- `training.md` says only `mhcflurry train processing-data` needs a source
  checkout; `validate-processing-data` runs from the installed package. Its
  opening keeps the instruction to always evaluate on held-out data.
- The landing page describes `training_provenance` accurately: it checks
  whether evaluation samples overlap any compared model's training data.
- The older allele-specific models are described as kept for reproducing
  earlier results, with the presentation bundle recommended for new work.
- The command reference states precisely which subcommands it lists without
  options.
- The Python tutorial and the API reference link to each other again.
- The Python tutorial's scanning example no longer calls its protein sequences
  "peptide sequences".
- `training.html#processing-training-data` and
  `commandline_tutorial.html#cli-next-steps` still resolve after the heading
  changes.
