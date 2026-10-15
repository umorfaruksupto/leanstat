# learnstat

An AI-assisted learning companion for statistical analysis in Python.

You write ordinary **pandas / NumPy / SciPy / Matplotlib** code. learnstat:

1. **Understands your dataset.** It profiles the data once and keeps a Markdown memory file of it.
2. **Suggests better approaches as you write.** For example, it flags a pie chart of 12 categories and suggests a sorted bar chart, explaining why.
3. **Explains itself** through `/describe`, `/alternatives`, `/learn`, `/memory` and `/fix`.
4. **Recovers from mechanical errors.** It fixes typos in column names, methods and keywords, plus missing imports and unclosed brackets, and shows you every change.
5. **Leaves analytical decisions to you.** Data-quality and statistical-method problems are explained but never "fixed" automatically.

It's not a new language. Everything you learn uses the real pandas, SciPy and Matplotlib APIs, so the skills carry over to the rest of the ecosystem.

## Install

```bash
pip install -e ".[all]"        # pandas, numpy, matplotlib, scipy, anthropic, ipython
pip install -e .               # core only: profiling, rules, recovery (no AI)
```

The AI features are optional. With the `anthropic` package and credentials set up (`ANTHROPIC_API_KEY` or `ant auth login`), explanations get tailored to your dataset, Claude reviews code with `/review`, and Claude can propose fixes for errors the rules can't handle. Without them, everything else still works offline. The default model is `claude-opus-5-5`; set `LEARNSTAT_MODEL` to choose another.

## Use it

### Jupyter

```python
%load_ext learnstat
import learnstat as ls
df = ls.read_csv("survey.csv")      # a normal DataFrame, now profiled into .learnstat/survey.md
```

```python
df["country"].value_counts().plot.pie()
# [!] Suggestion #1 (line 1): Pie chart with 12 categories
#     Issue:   `country` has 12 categories; slices this numerous are hard to compare by angle.
#     Instead: A horizontal bar chart of the counts, sorted by size.
#     Why:     Bars share a common baseline ...
#     How:     counts = df['country'].value_counts() ...
```

```
/describe        <- the concept, why it fits this data, when another method is better
/accept          <- puts the suggested code in a NEW cell; your cell is untouched
/reject only 3 countries matter here   <- recorded in memory, not suggested again
```

When a cell raises an error, learnstat diagnoses the failing statement. If it's a mechanical mistake, learnstat corrects it, re-runs only that statement and the ones after it, and prints the diff. A cell with a syntax error gets a corrected copy placed in the next cell for you to review.

### Command line

```bash
cd examples && python make_survey.py
learnstat profile survey.csv           # writes .learnstat/survey.md and prints it
learnstat check analysis.py            # suggestions only, nothing is executed
learnstat watch analysis.py            # re-checks every time you save the file
learnstat run analysis.py              # runs with suggestions and guarded recovery
learnstat repl --data survey.csv       # interactive, with /commands
```

### Plain Python

```python
import learnstat as ls
df = ls.read_csv("survey.csv")
ls.check("df['income'].mean()")         # suggestions without running
print(ls.command("/describe"))

s = ls.Session(confirm=lambda q: input(q + " [y/N] ") == "y")
s.ns["df"] = s.read_csv("survey.csv")
result = s.run("df.groupBy('group')['score'].median()")
print(result.final_code)                # df.groupby(...)... your source is never modified
```

## Commands

| Command | What it does |
|---|---|
| `/describe [id]` | Explains the latest (or given) suggestion, or the latest error: the concept, the purpose of the function, why it fits this data, and when something else is better |
| `/alternatives [id]` | Other ways to reach the same goal, with trade-offs |
| `/learn [topic]` | A short lesson with an exercise on your own data |
| `/fix` | The last error diagnosis; in interactive mode, a retry of the failed statement |
| `/memory [show\|note\|decide\|refresh]` | View or update the dataset's persistent profile |
| `/accept [id]`, `/reject [id] [why]` | Accept or reject a suggestion; the decision is saved in memory |
| `/suggestions`, `/review`, `/ask`, `/status`, `/help` | |

## Dataset memory

`ls.read_csv` profiles the data and writes two files to `.learnstat/`:

- `survey.md`: columns, types, missing values, likely column meanings (marked as guesses), distributions, relationships, data-quality issues, suggested analyses, assumptions, **your analytical decisions** and notes.
- `survey.json`: the machine-readable cache. It includes a content fingerprint, so the data is only re-profiled when it actually changes. Decisions and notes carry over when that happens.

## How error recovery stays safe

- Code runs **one top-level statement at a time**. When statement *k* fails, only statement *k* is corrected and retried, so earlier side effects are never repeated.
- Errors are classified as **syntax, name, attribute, argument, column, data-quality, statistical-method, environment** or other runtime errors.
- Only mechanical mistakes are corrected automatically, and only to a unique, high-similarity match.
- Before a fix runs, it must compile. It must not introduce file, shell or network operations, must not change which variables are assigned, must not add data-altering calls (`dropna`, `fillna`, `astype` and so on), and must not change the statistical method.
- AI-proposed fixes always need your confirmation. So does retrying a loop or block, since that could repeat side effects.
- Retries are bounded (`max_attempts`, 3 by default). If the same error comes back, recovery stops and gives you a concrete next step.
- Code that deletes or overwrites files, runs shell commands or opens network connections only runs after you confirm.
- **This is a policy layer, not a sandbox.** Your code runs in your Python process with your permissions, exactly as if you had run it yourself.

## Project layout

```
learnstat/
  profile.py      dataset profiling (no AI)
  memory.py       Markdown + JSON dataset memory
  advisor.py      AST-based, data-aware suggestion rules
  knowledge.py    offline explanations behind /describe, /alternatives, /learn
  recovery.py     error diagnosis, fix proposal, validation, bounded retry
  safety.py       detection of destructive side effects
  llm.py          optional Claude backend
  session.py      ties it together; read_csv/track/check/run
  commands.py     slash commands
  ipython_ext.py  %load_ext learnstat
  cli.py          learnstat profile|check|watch|run|repl
```

Add a rule by writing a function `(call, resolver, code) -> Suggestion | None` in `advisor.py`, appending it to `RULES`, and adding its concept to `knowledge.py`.

## Roadmap

- A language-server (LSP) frontend for suggestions while typing in VS Code and JupyterLab. The advisor is already editor-agnostic.
- More rules: multiple-comparison corrections, chi-square expected counts, normality checks before parametric tests, time-series plotting.
- Learning-progress tracking, so advice fades as the user masters a concept.

## Tests

```bash
pip install -e ".[dev]"
pytest
```
