# AGENTS.md

Homework and the group project of the ITMO "Efficient models" course, 2026. Every coding agent reads this file before the first task of a session, and the README of the folder it works in.

## Repository map

| Path | Content |
|---|---|
| `Efficient-Models-course-ITMO-2026/` | Course materials, a git submodule. Read only. |
| `hw1/`, `hw2/`, `hw3/` | Homework: code, notebooks, reports in `README.md`. |
| `project/` | Group project: knowledge distillation into a tiny encoder with 10% of the labels. `PLAN.md` is the plan, `README.md` the report. |

GPU work runs on Kaggle and Colab T4 instances; the local machine is a Mac without a CUDA GPU. Python projects use uv (`uv sync`, `uv run`).

## Working rules

1. Change only what the task needs. Do not reformat, rename, or rewrite code and text outside it.
2. Add no scripts, generated files, or templates outside the structure in the folder's README unless the user asks for them.
3. Commit or push only when the user asks. Commits carry no tool attribution.
4. Write repository text in English with American spelling. Messages to the user are in the language the user writes in.

## Writing rules

Every text in the repository (READMEs, plans, notebook markdown, slides, docstrings, comments, commit messages) and every message to the user follows these rules from the first draft.

### Content

5. Every sentence carries a specific, checkable statement. Delete a sentence that would fit any project.
6. Back each factual claim with a citation or a number. Give a number with its unit, model and size, dataset, split, baseline, hardware for timings, and the number of seeds with the spread.
7. Match the verb to the evidence: "shows" for a direct measurement, "suggests" for indirect evidence, "we expect" for an untested claim. Use at most one hedge per claim.
8. Attribute a gain to a component only when a controlled comparison isolates it; otherwise report the gain without a cause.
9. Where a source is silent, say so, with no speculation and no disclaimer such as "not widely documented".
10. A time or a result that was not measured is labeled as an estimate, with what it is estimated from.

### Structure

11. Give each paragraph, section, and document one point. The first sentence states it.
12. Order content by the logic of the subject. The chronology of the work goes into commit messages.
13. A heading over a result states the result: "KD on unlabeled texts closes 80% of the gap", not "Results".
14. Treat each topic in one place and link to it from elsewhere.
15. Give parallel content a parallel form.

### Terms

16. Define every term and abbreviation at its first use. Avoid an abbreviation that occurs fewer than three times.
17. Use one term for one concept throughout a text, with no synonym rotation: if it is "student", it is never also "small model" or "compact network".
18. Describe a model by its measured behavior and attribute no understanding or intent to it: "the student matched the teacher on 91% of test sentences", not "the student learned what the teacher knows".
19. Prefer plain words ("is", "has", "uses", "tried") to "serves as", "stands as", "represents", "boasts", "features", "utilizes", "leverages".

### Form

20. Put the action in the verb and the actor in the subject: "we trained the student on teacher logits", not "training of the student was performed". Use the passive only when the actor is unknown or irrelevant.
21. Cut words that carry no information: "in order to" becomes "to", "due to the fact that" becomes "because", "a number of" becomes the number.
22. Vary sentence length. Use colons, semicolons, and parentheses where they fit, no em dashes, the en dash only in numeric ranges, straight quotes only.
23. Use the past tense for what was done or observed and the present tense for how the code works and for what a table or figure shows.
24. Write explanations as prose; use a list only for parallel, separable items or for steps.
25. Use a table for runs, results, and other data with 3 or more rows; put smaller content into prose.
26. Headings are in sentence case, skip no level, and have text under them before the first subheading. No horizontal rules.
27. No bold or italics for emphasis (bold marks only a term at its definition), no bold label at the start of a list item, no emoji.
28. Leave no unfilled placeholders.

### Text about the text

29. A text states the current state of its subject. It contains no self-description ("This document describes", "Below is", "Here is", "In this section", "As mentioned above"), no change notes ("Updated", "Revised", "Previously", "(new)", "For simplicity", "for brevity"), and no justification of its own wording, format, or structure.
30. A plan states what will be done, with what, and what result counts as done. It does not explain why it is written the way it is or what it deliberately leaves out.
31. Commit messages: an imperative subject naming the concrete change ("Add k-center selection to kd.py"); the body gives the reason when it is not obvious. No "improve clarity", "ensure consistency", "refine", "streamline", no statement of what was preserved, and no claim that the change follows rules.
32. A message to the user opens with the answer or the result, then the decisions the user must make and the open questions. It narrates no steps and has no offers of further help, pleasantries, or apologies.

### Patterns of generated prose

These patterns mark text generated by language models (WikiProject AI Cleanup). Deleting the phrase alone leaves generic content in place: rewrite the sentence around its specific fact (rule 5), or delete it.

33. Closing recaps: "In summary", "In conclusion", "Overall", "Key takeaways", or a final paragraph that restates earlier text.
34. Negative parallelism: "not X but Y", "not only X but also Y", "not just X", "it is not X, it is Y", emphatic "Y rather than X".
35. Default triads: three adjectives, examples, or clauses where the content has a different number of items.
36. Doublets: paired near-synonyms where one word carries the meaning ("clear and concise", "challenges and limitations").
37. Significance inflation: "plays a crucial role", "a testament to", "marks a shift", "paves the way", "evolving landscape".
38. Participle tails and openers that comment on a fact: ", highlighting", ", underscoring", ", ensuring", ", reflecting", ", contributing to", "Building on this,".
39. Templated structure: one sentence frame repeated over content that is not parallel, such as paragraphs that each close with a one-line takeaway.
40. Promotional and overused words: "boasts", "vibrant", "profound", "showcase", "groundbreaking", "cutting-edge", "seamless", "game-changer", "comprehensive", "holistic", "additionally" (at sentence start), "align with", "bolster", "crucial", "delve", "enhance", "foster", "garner", "highlight" (verb), "intricate", "key" (adjective), "landscape" (figurative), "meticulous", "notably", "pivotal", "realm", "tapestry", "testament", "underscore", "valuable". Use "robust" only in its statistical sense.
41. Attribution padding: "experts argue", "studies show", "it is widely known" without citations.
42. Formulaic outlook: "Despite these challenges", "future outlook", "remains to be seen", "opens new possibilities".
43. Didactic disclaimers: "It is important to note", "It is worth noting", "may vary", "should be treated as X rather than Y".
44. Chat artifacts: "Certainly", "Of course", "Great question", "I hope this helps", "Let me know", "Would you like".
45. Rhetorical devices: rhetorical questions ("Why does this work?" as a heading or an opener), colon reveals ("The result? A gain."), sentence fragments for effect.
46. Tool residue: "contentReference", "oaicite", "turn0search", "[cite: N]", `utm_source` in URLs.

## Code rules

The first rule: match the surrounding code, its comment density, docstring format, naming length, and idioms. Experiment code here is a few flat scripts (`train.py`, `plots.py`, `kd.py`) with module-level constants and plain functions; keep it that way. The rules below apply where the code is silent. Rules 47-52 are the categories of [deslopify.ai](https://www.deslopify.ai/), which flags code generated by language models.

47. Self-evident comments: a comment says why, never what. No comment that restates the code or the name below it (`# load the model` over `load_model()`), no step banners (`# Step 1`, `# ===== Helpers =====`), no headers that repeat the file name. A comment is kept when it records a source (paper section, config value from the authors' repo), a unit, or a reason the code cannot express.
48. Emojis: none in code, comments, log lines, printed output, or notebook output.
49. Long-winded generated code: no jokey or chatty docstrings, no docstring that retells the branches of a function, no overcomplicated expression where a plain one exists (`x if x is True else False`, `not not x`), no wrapper whose body is one call.
50. Unnecessary checks: no conditions that are always true or false, no type checks or validation that the caller, the library, or the data format already guarantees, no defensive `None` checks on values that cannot be `None`, no retries on calls that do not fail transiently.
51. Duplicated logic: one computation lives in one place. No copy-pasted blocks that differ in one value; use a loop or a parameter. Two places that must stay in sync are a bug waiting to happen.
52. Deep nesting: return or `continue` early; more than 3 levels of indentation inside a function means the control flow can be flatter.
53. Error handling: catch only exceptions you can handle, by type. A catch handles the error or lets it propagate; never log and continue as if nothing failed, never bind an error and ignore it. No `try` around code that cannot fail.
54. Structure: no abstraction with a single caller (base classes, factories, config objects, registries) added for future extensibility. No unused imports, parameters, or variables; verify that imported names exist. Do not hand-roll what one library call does.
55. Naming: scale name length to scope (`i`, `x`, `b` in a loop, descriptive names at module level). No `Manager`, `Handler`, `Helper`, `Processor`, `Util` suffixes.
56. Completeness: never ship a stub with a real signature that returns a plausible fake value; implement it or state that it is not implemented. Cover the states that occur (a missing result file, an interrupted run, an empty subset).
57. Results: every reported number comes from a file the code wrote (`results/*.json`, `*.csv`), with the environment of the run recorded. No numbers typed into a README by hand that no file backs.

## Final pass

58. Before finishing, reread the changed text and code against these rules and fix every violation in place, without reporting it. To check a whole folder, run the `/deslopify` skill.
