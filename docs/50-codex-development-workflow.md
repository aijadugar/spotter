# How We Use Codex To Build Spotter

This is a short field note on how we used Codex while building Spotter, our small-model workout form
coach. Spotter takes a short exercise video and turns it into a structured form-review report: pose
analysis, exercise routing, rep counting, issue markers, annotated clips, and a grounded coach
summary.

That kind of product has a lot of moving pieces. It is not only a UI. It is not only a model. It has
data preparation, computer vision, small-model inference, deterministic rules, safety wording,
training scripts, deployment constraints, and docs. Codex helped because it could move across those
layers with the repo open, while still letting us stay in control of product direction.

Codex did not replace engineering judgment. It made the loop between idea, implementation, review,
and documentation much tighter.

## Why Codex Was Useful For This Project

The biggest advantage was context. A normal chatbot can answer a question, but Codex can inspect the
actual project: `app.py`, `src/spotter/`, `web/`, `scripts/`, `configs/`, `tests/`, and `docs/`.
That matters because the best answer for Spotter is usually not the most generic answer. It has to fit
the current pipeline.

For example, if we want to improve push-up feedback, the right starting point is not "write a new
fitness AI feature." The right starting point is:

- read the existing push-up analyzer
- read the shared rep counter and issue-marker helpers
- check the tests that define current behavior
- understand the JSON contracts used by the UI and coach summary
- then propose the smallest useful change

Codex is good at that kind of grounded work. It can keep the current codebase in view while it
brainstorms, implements, reviews, and documents.

## Brainstorming: From Vague Ideas To Scoped Work

Early in the project, many ideas started rough:

- Can the app explain bad reps better?
- Should unsupported exercises be rejected or forced into the closest label?
- How should we show confidence without making medical claims?
- What is the smallest useful coach summary model we can ship?

Codex was useful because we could ask it to brainstorm inside the project constraints. A good prompt
was not just "give me ideas." It was closer to:

```text
Read the current Spotter pipeline and brainstorm three ways to improve form feedback. Keep the ideas
compatible with the existing analyzer structure and avoid medical claims.
```

The result was more useful than a blank-page brainstorm. Codex could separate product ideas from
engineering tasks, identify likely files, and call out risk. That helped us avoid turning every idea
into a large rewrite.

The best brainstorming output usually had this shape:

- what the user problem is
- what the smallest version could be
- which files would change
- what tests would prove it works
- what wording needs human review

That is why Codex is good for early-stage product work: it can turn messy intent into a concrete
engineering path without pretending the path is risk-free.

## Deep Research: Faster Learning, Better Tradeoffs

Spotter touches several systems that change over time: Hugging Face Spaces, Modal, MediaPipe,
Gradio, small-model inference, and model publishing. We used Codex for deep research when we needed
to understand a tool before changing code.

The useful pattern was to ask Codex to separate facts from recommendations:

```text
Research the current deployment constraints that matter for this Gradio app. Prefer official
sources. Summarize the facts, explain what they mean for Spotter, then recommend changes only if they
are justified.
```

That made research actionable. We did not want a long pile of links. We wanted to know what affected
the project:

- Does this runtime support the dependency we need?
- Should this model run through hosted inference or local inference?
- Does this training job belong locally, in CI, or on Modal?
- Which setting affects GPU time, startup time, or reliability?

Codex was good here because it could connect research back to the repo. It could say, "this affects
the provider code," or "this belongs in the Modal script," or "this should be documented in the
training report."

We still treated research as research. Codex could make recommendations, but we converted those
recommendations into scoped implementation tasks before changing the project.

## Collaboration: Keeping The Team In The Same Thread

Codex was also useful as a collaboration tool. When multiple people touch a fast-moving project, the
hard part is often not writing code. The hard part is remembering why a change exists, what is still
untested, and what another teammate needs to know.

We used Codex to create handoff notes like:

```text
Summarize the current branch for another contributor. Include what changed, why it changed, how to
test it, and what still needs review.
```

That was especially helpful around training and deployment work. A branch might include a script
change, a config update, a docs update, and a model artifact note. Codex could inspect the diff and
turn it into a readable handoff.

The collaboration rule we kept was simple: Codex can help explain and organize work, but it should
not overwrite another member's changes. Before editing, we ask it to inspect `git status --short` and
read relevant diffs. That keeps the workflow respectful of everyone else's worktree.

## Code Review: A Fast Second Pass

One of the best uses of Codex was code review. Not a replacement for human review, but a fast second
pass before asking someone else to look.

The review prompt we used most often was direct:

```text
Review the current diff. Focus on correctness, regressions, missing tests, grounding, and user
safety. Put findings first with file and line references.
```

That framing matters. We did not ask Codex to nitpick style. We asked it to look for things that
could break the product:

- a pipeline contract changed but the UI still expects the old field
- a fallback path no longer works when a model provider fails
- a test fixture covers only the happy path
- a coach summary can say more than the structured evidence supports
- a deployment setting works locally but not on Hugging Face Spaces

Codex is good at review because it can inspect related files quickly. If a change touches
`src/spotter/steps/coach_summary.py`, it can also check the verifier, fallback summary, provider
tests, and docs. That is the kind of cross-file attention that catches practical regressions.

## Implementing UI And Code

For implementation, Codex was most helpful when the task was scoped. "Improve the app" is too broad.
"Update the result view to show summary provider metadata and add a focused test" is a good Codex
task.

For Python pipeline work, we ask Codex to follow existing project structure:

- pipeline steps live under `src/spotter/steps/`
- exercise logic lives under `src/spotter/exercises/`
- shared contracts live in `src/spotter/contracts.py`
- training and publishing workflows live in `scripts/` and `configs/`
- behavior should be covered in `tests/`

For UI work, we ask it to inspect both the Gradio entrypoint and static assets:

- `app.py`
- `web/index.html`
- `web/app.js`
- `web/report.js`
- `web/styles.css`

The strongest Codex implementation loop looks like this:

1. Read the relevant files.
2. Explain the smallest safe change.
3. Make the edit.
4. Add or update focused tests.
5. Run the relevant checks.
6. Summarize what changed and what remains uncertain.

That loop is where Codex feels different from autocomplete. It is not only producing lines of code.
It is helping maintain the whole change: code, tests, docs, and verification.

## Plugins: Bringing The Right Tool Into The Same Flow

One thing that made Codex more effective was using plugins for tasks that needed more than plain code
editing. The value was not "more tools for the sake of tools." The value was staying in one
development flow while Codex used the right capability at the right time.

For Spotter, the most useful plugin pattern was UI verification. When we changed the app interface,
Codex could edit the frontend code, start the local app, open it in a browser, inspect the result,
and then come back to the code with a concrete fix. That is much better than only reading CSS and
guessing whether the page looks right.

Plugins also helped with artifact-heavy work. Spotter has reports, model-card style docs, demo notes,
and training writeups. When the output is a document, presentation, spreadsheet, screenshot, or PDF,
it is useful for Codex to work with the artifact directly instead of treating everything like raw
text.

The practical lesson was simple: use a plugin when the task has a real environment or artifact to
inspect.

- For UI work, use browser inspection instead of trusting code alone.
- For docs and reports, use document-aware workflows when layout or structure matters.
- For product design work, use design-oriented workflows before jumping into implementation.
- For generated artifacts, ask Codex to render or verify the result when possible.

That made Codex feel less like a detached assistant and more like a teammate sitting inside the same
workspace.

## Skills: Turning Good Habits Into Repeatable Playbooks

Skills were useful for a different reason. A plugin gives Codex a capability. A skill gives Codex a
way of working.

In this project, we used skills as repeatable playbooks for work that needed a consistent standard.
For example, documentation should not be a random dump of notes. It should have a clear audience,
scope, and structure. UI work should not only "compile"; it should be checked for layout, responsive
behavior, and product fit. Code review should start with bugs and regressions, not style opinions.

Skills helped encode those expectations. Instead of re-explaining the standard every time, we could
ask Codex to use the relevant skill and then let it follow that workflow:

- documentation skills for clear project docs, reports, and handoff notes
- frontend/design skills for UI changes that need visual quality and responsive behavior
- code review behavior for focused review comments and missing-test analysis
- product or research skills when we needed to compare options before implementation

The important habit was to invoke the skill before the work starts. That makes Codex read the right
instructions first, then inspect the project, then act. The result is more consistent than asking for
a one-off answer each time.

For a fast project like Spotter, that consistency mattered. We were moving between model training,
UI, docs, deployment, and tests. Skills helped keep the quality bar stable while the task type kept
changing.

## Automation: Turning Repeated Work Into Scripts

Spotter has repeated workflows: running fast tests, preparing data, training routers, training coach
summaries, publishing artifacts, and keeping docs in sync. Codex helped us turn some of those
manual steps into explicit scripts and checklists.

Automation is a good Codex task because the desired behavior can be made concrete:

```text
Add a script that runs the fast Spotter validation checks before a PR. Reuse existing commands, avoid
network-dependent steps, and document how to run it.
```

The important part is that automation should be boring. It should log clearly, fail clearly, and avoid
surprising side effects. For this project, anything involving credentials, model uploads, dataset
publishing, or GPU spend still needs human approval.

Codex is good at automation because it can inspect how the project already runs. It can reuse
`uv run pytest`, `uv run ruff check .`, Modal scripts, existing config files, and docs instead of
inventing a parallel workflow.

We also used Codex to decide what should not be automated. Some actions are too expensive or risky to
run silently: uploading a model, publishing a dataset, spending GPU time, changing public demo
behavior, or rewriting safety wording. For those, the better automation is a checklist or a command
with an explicit approval step.

That split made automation more useful:

- automate local checks that are cheap and repeatable
- script data and training setup when the inputs and outputs are clear
- document manual approval points for publishing and public claims
- use reminders or handoff notes for follow-up work that should not block a coding session

The best automations were small. A good script saved a few minutes every time and made failure
obvious. A good checklist prevented a risky release mistake. Codex helped build both.

## How Plugins, Skills, And Automation Fit Together

The most effective Codex workflow combined all three.

Plugins gave Codex access to the working surface. Skills gave it the right operating style.
Automation made the repeated parts cheap.

For example, a UI change could flow like this:

1. Use a frontend or product-design skill to frame the change.
2. Ask Codex to inspect `app.py` and `web/`.
3. Implement the smallest UI update.
4. Use the browser plugin to open the local app and check the rendered result.
5. Run focused tests or linting.
6. Update docs or write a handoff note.

A training workflow looked different:

1. Use Codex to research or review the training goal.
2. Inspect `scripts/`, `configs/`, and the relevant training docs.
3. Update the script or config in a scoped way.
4. Automate only the safe local checks.
5. Keep model upload, dataset publishing, and GPU-heavy runs behind human approval.
6. Record metrics and artifact paths in the docs.

That is where Codex became especially effective. It was not one magic prompt. It was a repeatable
system: choose the right playbook, use the right tool, automate the boring part, and keep human
judgment on the decisions that matter.

## What Makes Codex Good

For this project, Codex was good for eight practical reasons.

First, it works with the real repo. It can read the current files, not just guess from a description.
That makes its suggestions more grounded.

Second, it moves across layers. Spotter needs Python, web UI, ML scripts, configs, tests, and docs.
Codex can connect those pieces in one task.

Third, it is good at turning ambiguity into a plan. When an idea is vague, Codex can propose options,
tradeoffs, affected files, and a smallest useful version.

Fourth, it is good at review. It can look at a diff and check related files faster than a human can
manually scan the whole repo.

Fifth, it helps preserve momentum. Instead of stopping to remember the exact test command, docs
location, or helper API, we can ask Codex to inspect and continue.

Sixth, it improves documentation while the context is still fresh. After implementing a change, Codex
can update the relevant docs and write a handoff note before details are forgotten.

Seventh, plugins let it inspect real outputs. That is important for UI, documents, generated
artifacts, and local app behavior.

Eighth, skills and automation make the workflow repeatable. The team does not have to rebuild the
same process from memory each time.

## What We Still Keep Human-Owned

The biggest lesson is that Codex works best when responsibility stays clear.

Humans still own:

- product direction
- user safety language
- fitness and health-related claims
- dataset choices and licensing judgment
- public model and dataset publishing
- final code review and merge approval
- whether a feature is actually useful to users

Codex helps us move faster, but it does not decide what Spotter should be. That distinction matters a
lot for a product that gives workout feedback. The app should be grounded in evidence, and the
development process should be grounded too.

## Our Current Codex Workflow

The workflow we settled into is simple:

1. Use Codex to inspect the repo and understand the current shape.
2. Brainstorm or research with project constraints in view.
3. Pick a small, human-approved direction.
4. Ask Codex to implement the scoped change.
5. Ask Codex to review the diff.
6. Run tests, linting, local app checks, or browser checks.
7. Update docs and write a handoff note.

That workflow made Codex valuable throughout the build. It helped us think, research, collaborate,
review, implement, and automate without turning the project into a black box.

The short version: Codex is good because it compresses the distance between intent and verified
change. For Spotter, that meant more time spent on product judgment and less time lost to mechanical
work, context switching, and stale documentation.
