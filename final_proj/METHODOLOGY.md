# Algorithm specification and baseline fidelity

**Current implementation is V6.** This document describes the original formulation and its baseline rationale. The consolidated README explains the current depth and diversity terms, judging cost, numeric expressions, complete-solution incumbent, reopening rule and final answer selection. [Current search methodology](README.md#research-question-and-approach)

## Scope

The project studies a heuristic for selecting **intermediate reasoning states**.
It does not use the earlier repair-versus-explore controller or train a policy to
optimize expected accuracy gain. The proposed method is bounded best-first search,
not A* with admissibility, completeness or optimality guarantees.

## State, transition and checks

A node stores the original question through the engine, an ordered reasoning path,
parent/node IDs, depth, generated path tokens, evaluator scores, completion flag
and candidate numeric answer. A transition asks the model for exactly one new step.
Only the selected node's children are generated; the whole tree is never generated
up front. Strict JSON schemas constrain outputs. Python assigns positional step
IDs and derives dependencies from expressions. Redundant legacy metadata is
normalized and logged. A bare assignment to the current canonical step ID may
be stripped and logged before arithmetic validation; assignments to other names
remain invalid. Generation schemas restrict expression characters and one-step
prompts enumerate available references; unknown/forward references and unsafe expressions are
rejected uniformly across methods.

Each step names a quantity and units and contains an arithmetic expression with
references to prior canonical step IDs. A restricted AST interpreter supports constants,
step references and + - * /. Exact Fraction arithmetic recomputes values for every
method. No generated Python code is executed. Arithmetic correctness is not proof
of correct semantic interpretation. A complete solution must calculate the requested
quantity in its last step. The model no longer supplies an independent answer field.

## Five-factor heuristic

For unfinished state s:

    H(s) = wQ*Q(s) + wP*P(s) - wD*D(s) - wC*C(s) + wV*V(s)

Weights are finite, nonnegative and sum to one. Features lie in [0,1]. Defaults
are Q=.35, P=.30, D=.05, C=.15, V=.15; they have not been tuned on GSM8K results.

- **Q: existing reasoning quality.** The same model, in a separate evaluation
  prompt, assesses logical soundness and relevance of the existing path.
- **P: future-success confidence.** The evaluator separately estimates whether
  continuing the path under the stated depth/call limits can reach the answer.
  This is an **uncalibrated model judgement**, not a verified probability.
- **D: depth / maximum depth.** Modest depth penalty; it can suppress necessary
  long reasoning, which is why a no-depth ablation matters.
- **C: estimated additional generation cost.** Estimate next prompt tokens as
  UTF-8 prompt bytes / 4 + 32 template tokens; estimate next output tokens using
  an exponentially weighted average of observed single-step generation lengths
  (initially the step token cap). Divide their sum by a fixed 1200-token reference
  and cap at 1. This is a proxy for one next child generation, not exact FLOPs or
  total tree-completion cost. It excludes the subsequent judge from the ranking
  proxy; all actual judge tokens ARE charged in evaluation metrics. Prior path
  spend is recorded but not penalized again as a recoverable cost.
- **V: lexical novelty.** 1 minus maximum TF-IDF cosine similarity to previously
  expanded paths. Text includes quantity descriptions and expressions. IDF is
  fitted locally on current frontier candidates and expansion history, with
  smoothed IDF and log term frequencies. An empty history yields novelty 1.
  This is not a claim of semantic or statistical independence.

In v3, the default diversity feature is V_gated = V * sqrt(Q*P). The same
five weights are retained. Low-confidence novelty therefore contributes less;
a zero Q or P removes the novelty bonus. This can also suppress paths that the
judge undervalues: it is an unvalidated heuristic change, not a guarantee. Raw and
gated diversity are both logged. `quality_gated_diversity=false` restores the
ungated formula with the same shared generation fixes for a controlled ablation.

Q and P are cached after evaluation; they may become less representative as the
remaining budget changes. C and V are refreshed before selection, and the heap is
rebuilt to avoid stale diversity priorities. Exact path duplicates are removed
conservatively; paths are not merged merely because their wording is similar.

The evaluator first records coverage, interpretation, units and completion checks
as pass/fail/uncertain/pending, then gives Q/P scores. Unfinished paths may have
pending coverage without being erroneous. These checks are model judgments, not
formal verification, and do not impose an additional hard score gate.

The evaluator scores a small batch of sibling states in one request. Both Q and P
are requested for all search methods, even Q-only controls, to hold evaluation
overhead/prompt content fixed when comparing state-selection rules. Invalid
evaluations receive zero scores and a failure flag, never automatic confidence 1.

## Search loop

1. Initialize a frontier containing the root.
2. Refresh frontier features and select the highest-priority state.
3. Generate up to two one-step children with the shared generator.
4. Check expressions, reject malformed states and remove exact duplicate paths.
5. Evaluate children, store completed candidates, and enqueue eligible unfinished
   children below the depth limit.
6. Prune the frontier by the method's ranking if it exceeds its capacity.
7. Repeat until no frontier remains, the expansion limit is reached, or a common
   call/token/context guard stops additional inference.

Completed paths do not consume more expansions. All search methods use the same
final-answer selector: highest Q among evaluated completed candidates, then first
generated as a tie-break. If only unevaluated completions exist due to exhaustion,
use the first valid completion and log the budget stop. No valid completion means
no prediction and an incorrect result. Gold labels never reach the policy.

## Comparison methods

1. **CoT:** one complete structured solution; calculator-assisted.
2. **Self-Consistency:** six independent complete solutions; plurality voting after
   numeric normalization, first-generated tie-break. No judge or early stopping.
3. **ToT breadth/beam:** breadth-wise expansion with beam width two, retaining the
   highest-Q children at each depth. This is a bounded beam adaptation, not an
   exhaustive breadth-first traversal or original ToT benchmark reproduction.
4. **Prompted self-evaluation stochastic beam:** the same level-wise expansion,
   but sample survivors without replacement with weights proportional to
   exp(Q / temperature), temperature .3. It does NOT implement the published
   generation-likelihood/correctness-logit objective and must remain labelled
   as a prompted-score adaptation. It uses the same evaluator as the controls.
5. **Quality-only best-first:** exactly the proposed best-first engine, except its
   state priority is Q alone. This isolates the combined heuristic from traversal.
6. **Proposed five-factor best-first:** ranks by H above.

All methods share the model, arithmetic tool, output format and global ceilings.
Search methods share generation, evaluation, maximum depth, branch factor and
expansion limit. Independent call seeds include method IDs, so candidate samples
are not identical across methods. Rotate run order; use repeated seeds to quantify
sampling variance. Full reproducibility is limited by backend/GPU nondeterminism.

## Structural recovery and label audit

Each generation can retry once after a malformed response, invalid expression or
missing required completion in full-solution mode. The same rule applies to all
six methods. Validation feedback contains no gold answer, aliases are never
guessed, and all attempts consume the existing call/token budgets. A valid but
semantically wrong answer does not trigger recovery. Infrastructure failures still
require explicit resume handling. Retry and metadata-normalization events are saved.

The evaluator preserves original labels and also reports separately reviewed
accuracy using `data/label_corrections.json`. One pilot label was reviewed by
arithmetic (430+750+300+700=2180 rather than the original 2280). The correction
requires exact question and original-label matches, is frozen in each manifest,
and cannot enter the Engine. Both sets of accuracy intervals and paired differences
are reported. Existing runs are neither changed nor retroactively corrected.
Because two test questions informed these fixes, disclose their exposure in any
report; they are diagnostic examples, not independent evidence of improvement.

## Accounting and inference safety

Ollama supplies measured prompt/output token counts. Record all successful judge,
generation and malformed-response costs. Replayed successful calls retain their
original costs. Error records survive retries and trigger unknown-cost warnings;
never claim exact efficiency when failed requests have unreported token usage.
Warmup is stored separately. Wall time includes backend/cache overhead. The runtime
logs local execution wall time as well, but it is different on replayed attempts.

A conservative UTF-8-byte context guard avoids knowingly overflowing context.
Its reserved budget can be larger than the actual token count and can stop a
method early; stop reasons are reported. No hidden extra final-answer calls are
made outside the method budgets. Common caps do not imply equal actual compute.

## Evaluation

Use the bundled 100-question test split only after development. Compare exact
normalized numeric final answers (decimals and fractions are canonicalized).
Do not filter questions based on whether the proposed method succeeds. Compare
all methods on complete blocks of the same questions. Report accuracy, calls,
tokens, time, expansions, maximum reached depth, duplicates and formatting failures.
Use a development-only weight sweep and remove-one-factor ablations. Compare the
same search engine with Q alone before attributing gains to the new score.

Bootstrap paired accuracy differences over questions; repeated seeds belonging
to the same question must stay together. Default folders hold one seed each;
combining runs requires an explicit grouped analysis. Wilson intervals are
single-seed descriptive intervals. With 100 questions uncertainty is substantial.
No SOTA, global novelty, calibrated-probability or equivalence claim is established
by this implementation or its synthetic unit tests.

## Relation to the research papers

- Thought of Search generates reusable search components; its soundness claims
  do not transfer automatically to our natural-language steps. We borrow local
  search control/checking, not its full code-generation method:
  https://arxiv.org/abs/2404.11833
- ChemReasoner uses domain feedback for heuristic-guided exploration. Arithmetic
  checks here are a limited feedback signal, not its chemistry reward model:
  https://arxiv.org/abs/2402.10980
- Tree of Thoughts: https://arxiv.org/abs/2305.10601
- Self-Consistency: https://arxiv.org/abs/2203.11171
- Self-Evaluation Guided Beam Search (original formulation differs as stated):
  https://arxiv.org/abs/2305.00633
- GSM8K data: https://github.com/openai/grade-school-math

Blend-ASC, CAPS and Repair-or-Explore are not included in this corrected project.

V3 was informed by inspecting 18 test questions, chiefly structural failures.
Report this exposure; use separate development data for settings and fresh held-out
questions for confirmatory claims. Shared formatting improvements must not be
attributed solely to the heuristic. See the [pilot results and failure analysis](README.md#what-the-earlier-gsm8k-pilot-revealed).
