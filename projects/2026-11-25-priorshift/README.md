# priorshift: fix a classifier when the class mix changes and nobody labels production

A classifier learns two things at once: what each class looks like, p(x | y), and how common each
class is, p(y). When it is deployed somewhere the **class mix is different**, the first part still
holds but the second is wrong, and the model keeps predicting with the old base rates. This is
**label shift** (also called prior or prevalence shift). Some examples:

* a fraud model trained at 0.5% fraud, running during an attack at 3%
* a triage model moved from a general clinic to a specialist one
* a form reader trained on balanced digits, now reading a postcode field that only contains 0–3

The obvious fixes don't work well. You can't retrain, because there are no labels yet. Counting
the model's own predictions ("classify and count") is biased toward the training mix, because a
model that confuses 3s with 8s keeps predicting about as many 8s as before. And "always re-weight
by the estimated mix" makes things *worse* when nothing has changed.

priorshift estimates the new class mix from **unlabeled predictions only**, corrects the
probabilities with Bayes' rule, and adapts only when a test says the mix really moved. The
benchmark measures each step against no adaptation and against an oracle that knows the true mix.

## What it does

```
              train split (50%)                     val split (25%)                    pool (25%)
                     │                                    │                       never used for fitting
                     ▼                                    ▼                                │
   StandardScaler + LogisticRegression ──► raw p(y|x) ──► BCTS calibration                 │
                                                          (temperature + per-class bias)   │
                                                          hard & soft confusion matrices   │
                                                          reference prior π                │
                                                                    │                      ▼
                                                                    │        label-shifted batch, mix q unknown
                                                                    ▼                      │
                                         calibrated p(y|x) on the batch ◄──────────────────┘
                                                    │
                    ┌───────────────────────────────┼─────────────────────────────┐
                    ▼                               ▼                             ▼
          prior estimators                 shift test (detect.py)        posterior correction
          cc / pcc         (naive)         LR test of q̂ vs π, χ²(K−1)    p_new ∝ p(y|x) q̂_y / π_y
          acc / pacc       (confusion      + min effect size (TV ≥ 0.05)  with add-one smoothing
                            inversion on                                   of q̂
                            the simplex)
          em               (Saerens MLE)
```

| module | contents |
|---|---|
| `priorshift/estimators.py` | CC, PCC, ACC (BBSE-hard), PACC (BBSE-soft), EM/MLLS, simplex-constrained confusion inversion, Bayes posterior correction, add-one smoothing |
| `priorshift/calibration.py` | bias-corrected temperature scaling (BCTS) with an analytic gradient, NLL, ECE |
| `priorshift/detect.py` | likelihood-ratio shift test with an effect-size gate |
| `priorshift/data.py` | bundled sklearn digits, optional 2× downsampling, stratified seeded 50/25/25 split, label-shifted batch sampler |
| `priorshift/model.py` | fits classifier, calibrator and confusion matrices; save/load with joblib |
| `priorshift/evaluate.py` | simulated deployment benchmark and Markdown/JSON report |
| `priorshift/cli.py` | `train`, `estimate`, `make-batch`, `benchmark` |

## Setup

Python 3.11+ (tested on 3.11.18 with numpy 2.4.6, scipy 1.17.1, scikit-learn 1.9.1).

```bash
cd projects/2026-11-25-priorshift
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
```

No downloads. The digits dataset ships inside scikit-learn.

## Usage

```bash
# 1. fit classifier + calibration + confusion matrices (seed fixes the split)
python -m priorshift train --out model.joblib --seed 0

# 2. estimate the mix of an unlabeled batch (CSV with columns p0..p63, raw 8x8 pixel values 0-16)
python -m priorshift estimate --model model.joblib --batch samples/batch_form_b.csv --out preds.csv
#    --labels samples/batch_form_b_labels.csv   adds a check against true labels (not used for estimation)
#    --method em|acc|pacc|cc|pcc                 default em
#    --no-smooth  --alpha 0.01  --min-tv 0.05

# 3. make your own shifted batch from the held-out pool of split seed 0
python -m priorshift make-batch --priors 0,5,0,0,0,0,0,1,0,0 --n 400 --out my_batch.csv --labels my_labels.csv

# 4. full benchmark (≈ 20 s) -> results/REPORT.md, results/report.json
python -m priorshift benchmark --out results            # --quick for a 2-seed smoke run
```

### Sample inputs and expected outputs

`samples/` contains two 300-row batches drawn from the held-out pool of split seed 0, their
labels (used only for the check), and `expected_outputs.json` with the full `estimate` output.
`scripts/make_samples.py` regenerates all of it, and `tests/test_pipeline.py` checks that the CLI
reproduces it.

| batch | true mix | estimated mix (EM) | shift test | accuracy before → after |
|---|---|---|---|---|
| `batch_form_b` (postcode field: only 0–3) | .317 .280 .230 .173 0 … 0 | .309 .271 .229 .186 0 0 0 0 .006 0 | p = 3.9e−89, TV 0.59 → adapt | 0.913 → 0.957 |
| `batch_balanced` (matches training) | ≈ uniform | TV 0.08 from π | p = 0.59 → leave alone | 0.907 → 0.907 |

On `batch_form_b`, the uncorrected model assigns 15 rows to 8 and 9, classes that do not occur in
the batch. After correction it assigns 2.

## Results (measured, `results/REPORT.md`)

Setup: 5 split seeds × 5 scenarios × batch sizes {100, 500, 2000} × 20 repeats = 1,500 batches.
The model is logistic regression on 4×4 downsampled digits, with validation accuracy 0.89–0.92 and
pool accuracy 0.89–0.91. Scenarios are `no_shift`, `surge` (one class at 40%), `missing` (3
classes absent), `dirichlet_1` and `dirichlet_0.3` (random mixes, the second very skewed). "TV" is
the total-variation distance between the estimated mix and the batch's realised mix.

**All 1,200 shifted batches combined:**

| variant | prior TV error | accuracy | NLL | batches where NLL beats no adaptation |
|---|---|---|---|---|
| none (training priors) | 0.378 | 0.8992 | 0.3230 | – |
| cc | 0.059 | 0.9163 | 0.2589 | 94.0% |
| pcc | 0.078 | 0.9146 | 0.2631 | 94.7% |
| acc | 0.055 | 0.9172 | 0.3286 | 76.3% |
| pacc | 0.049 | 0.9175 | 0.3083 | 78.1% |
| em | **0.040** | 0.9180 | 0.2653 | 88.9% |
| acc + smooth | 0.058 | 0.9176 | 0.2553 | 91.3% |
| pacc + smooth | 0.053 | 0.9178 | 0.2530 | 92.3% |
| **em + smooth (gated)** | 0.046 | **0.9180** | **0.2509** | 93.3% |
| oracle (true batch mix) | 0 | 0.9253 | 0.2283 | 96.9% |

What the numbers show:

* **EM gives the best mix estimate** (TV 0.040, against 0.059 for classify-and-count). It closes
  about 72% of the accuracy gap between no adaptation and the oracle (0.8992 → 0.9180 vs 0.9253)
  and about 76% of the NLL gap once smoothed.
* **The mix estimate alone is not enough.** Unsmoothed ACC, PACC and EM sometimes put exactly 0
  on a class that is rare but present. Those rows then get near-zero probability for their true
  class, so ACC's mean NLL (0.3286) is *worse than not adapting*. Add-one smoothing,
  (n·q̂ + 1)/(n + K), is a fixed rule that was not tuned on test data. It fixes this and costs a
  little TV on small batches. In the `missing` scenario smoothing is slightly worse (n = 100: NLL
  0.2590 unsmoothed vs 0.2661 smoothed), because there the zeros are real.
* **Adapting when nothing changed hurts**, and gating prevents it:

  | no_shift, NLL | n = 100 | n = 500 | n = 2000 |
  |---|---|---|---|
  | none | 0.3286 | 0.3229 | 0.3211 |
  | em + smooth, always | 0.3475 | 0.3317 | 0.3275 |
  | em + smooth, gated | 0.3293 | 0.3234 | 0.3240 |

  On every shifted scenario the test fired on 100% of batches, so gating cost nothing there.
* **The shift test's false-alarm rate is too high on large batches.** On `no_shift` the alarm
  rate is 4.0% at n = 100, 3.0% at n = 500 and **30%** at n = 2000, against a nominal α = 1%. The
  effect-size gate lowered it only from 40% to 30%. Two causes add up. First, small calibration
  errors become "significant" once n is large. Second, a 2,000-row batch must re-sample the
  ~450-image pool with replacement, so the pool's particular errors repeat. Treat the p-value as a
  ranking signal, not a calibrated error rate, at large n.
* On full-resolution 8×8 digits (`--factor 1`, accuracy ≈ 0.96) the same ordering holds with
  smaller gains: accuracy goes from 0.9623 to 0.9713 with em + smooth, and NLL from 0.1201 to
  0.0921 (oracle 0.0858).

## Design decisions

* **No leakage by construction.** The classifier sees only `train`. Calibration, confusion
  matrices and the reference prior come only from `val`. Batches come only from `pool`. Estimators
  take predicted probabilities as their only per-batch input, and labels are used afterwards to
  score. `test_fit_never_uses_the_pool` replaces the pool with noise and checks that every fitted
  quantity is identical.
* **Calibrate before EM.** EM is the maximum-likelihood estimate only if p(y|x) is calibrated.
  BCTS (temperature plus per-class bias, Alexandari et al. 2020) is fitted on `val` and lowers
  validation NLL for all 5 seeds. A test checks that it recovers a known temperature of 3.
* **Constrained inversion.** ACC/PACC solve C q = μ by non-negative least squares with a
  sum-to-one row, rather than q = C⁻¹μ. That inverse returns negative "prevalences" on small
  batches (a test shows this).
* **Two-part trigger.** The trigger requires both statistical significance and a minimum effect
  (TV ≥ 0.05). A significance test alone fires on harmless differences once batches are large.
* **Smoothing is a rule, not a tuned knob.** A pseudo-count of one per class goes to zero as n
  grows. Tuning a floor on the benchmark would have leaked test information into the method.
* **Score against the realised mix.** A 100-row batch drawn at the training priors still has a
  realised mix about 0.12 TV away from them. The oracle and the TV metric use the batch's actual
  label counts.
* **Downsampled digits by default.** At 96% accuracy the class mix barely matters. The 4×4
  version (≈ 90%) is closer to a real model with real confusions. `--factor 1` is available and
  reported above.

## Tests

26 tests (`python -m pytest -q`):

* **estimators**: run in a synthetic Gaussian world where the Bayes posteriors are known in
  closed form. Checks that EM, ACC and PACC recover the true mix, that CC and PCC are biased toward
  training, that the posterior correction equals the exact target posterior, and that the simplex
  projection, constrained inversion, smoothing and input validation behave correctly.
* **calibration and detection**: BCTS recovers a temperature distortion and a class bias. The LR
  test's false-alarm rate is within 1–11% at nominal 5% with exact posteriors, the test fires on a
  real shift, and the effect-size gate works.
* **pipeline**: splits are disjoint, stratified and reproducible. Fitting is independent of the
  pool, the model round-trips through save/load, and the quick benchmark gives the expected
  ordering. The CLI reproduces `samples/expected_outputs.json`, writes predictions and rejects bad
  CSVs.

## Limitations

* **Label shift is an assumption, not something checked.** If p(x|y) also changes (new
  handwriting styles, a new fraud pattern), these estimators can confidently report a wrong mix.
  Nothing here detects covariate or concept shift.
* The benchmark simulates shift by resampling one small public dataset. The pool holds about 45
  images per class, so batches of 500 and 2,000 repeat images. Results on real production drift
  will differ.
* The shift test's false-alarm rate exceeds nominal at n = 2000 (30% vs 1%), as described above.
* Only one model family (logistic regression) was evaluated. Tree ensembles, for example, are
  often worse calibrated, and EM depends on calibration.
* The training mix of digits is close to uniform. A heavily imbalanced training set (such as
  fraud) makes rare-class estimation harder and is not covered.
* Batch-level only: no online or streaming estimate and no confidence interval on q̂.
