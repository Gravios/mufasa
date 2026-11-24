# 9. Behaviour classifiers

A behaviour classifier learns, from frames you have labelled, to recognise a behaviour (rearing,
grooming, attack, …) in every frame of every video. Mufasa trains one random-forest classifier per
behaviour from the features ([chapter 8](08-features.md)).

```text
Manage classifiers ─► Annotation ─► Train ─► Validate ─► Inference ─► Analysis · Visualizations
(which behaviours)    (label frames)  (fit)  (check on     (every video)  (bouts, time, plots)
                                              new video)
```

Each step is its own page in the sidebar.

## Manage classifiers

**Manage classifiers** is a table of the project's behaviours:

| Column | What it is |
|---|---|
| Name | the behaviour's name (e.g. `Rear`) |
| Key | its hotkey in the frame labeller |
| Info | whether it has a trained model |
| Delete (×) | remove the behaviour from the project |

*+ Add classifier* adds a behaviour. Behaviours must exist before you can label them.

## Annotation

### Frame Labeling

**Annotation › Frame Labeling** opens the frame labeller on a video:

| Mode | What it does |
|---|---|
| New labeling | start from no labels |
| Continue labeling | load this video's labels and carry on |
| Pseudo-labeling (seed from classifications) | start from a classifier's predictions and correct them, which is much faster than labelling from nothing |

The table lists each video and whether its labelling is finished. *Label* opens the labeller and
*Refresh* re-reads the table.

In the labeller, labels are painted over frames as the video plays:

| Key | What it does |
|---|---|
| a behaviour's hotkey | turn continuous labelling of that behaviour on or off. While on, every frame played is labelled |
| `Delete` | turn continuous unlabelling of the active behaviour on or off |
| `←` / `→` | play backwards or forwards (press again to pause) |
| `Space` | pause or resume |
| `Shift`+`←` / `→` | jump 10 frames |
| `↑` / `↓` | play faster or slower (×1.25) |
| `Page Up` / `Page Down` | jump to the previous or next start or end of a labelled span |
| `Ctrl`+`S` | save |

Tick *Mark video complete* when a video is done; the table shows it. Labels are saved to
`derived/labels/<video>.parquet`.

**How much to label.** Label every frame of the behaviour in the stretches you label, and leave
the rest of those stretches unlabelled (which counts as "not the behaviour"). A few minutes of
video per behaviour, across several animals and sessions, is a reasonable start. Add more where
the classifier is wrong.

### Targeted annotation clips

**Annotation › Targeted annotation clips** marks the stretches of a video worth labelling (where
the behaviour happens, and where it is easily confused) and labels only those.

### Third-party annotation import

**Annotation › Third-party annotation import** brings in labels made elsewhere: BORIS, BENTO,
EthoVision, Solomon Coder, DeepEthogram.

| Option | What it does |
|---|---|
| Source directory | the folder of the other program's files |
| Third-party app · File format | which program, and its export format |
| On format mismatch | *WARNING* (skip the file and say so) or *ERROR* (stop) |
| *Write merge log* | log what was merged into what |

Recipe R7 in [`../workflow_recipes.md`](../workflow_recipes.md) covers each program's quirks.

### Review classifier predictions

**Annotation › Review classifier predictions** › *Review* steps through the classifier's predicted
bouts one by one. Mark each *valid*, *invalid* or *unsure*. This checks a classifier, and shows the
frames worth adding to the labels. The ratings are saved to `validation_results/<video>.csv`.

### Reports

**Annotation › Reports** has two actions:

* *Extract labelled frames*: export them as images, with a downsample factor, an image format and
  a greyscale option;
* *Annotation counts*: count labelled frames and bouts per behaviour (*Split counts per video* for
  each video separately).

## Train

**Train › Train classifier** trains a random forest for the behaviour chosen under *Classifier (target behavior)*, from the features of the labelled frames:

| Option | What it does |
|---|---|
| RF estimators | the number of trees (more is better, up to a point, and slower) |
| Max features · Criterion · Min samples per leaf | the trees' settings |
| Test size · Train-test split type | how much labelled data is held out to test on (0.1–0.5), and how: *FRAMES* or *BOUTS*. *BOUTS* is more honest, since neighbouring frames are alike |
| Under-sample setting / ratio · Over-sample setting / ratio · Class weights | balance rare behaviours against the many frames without them: random under-sampling, SMOTE or SMOTEENN over-sampling, or *balanced* class weights |
| Evaluations | *Classification report*, *Feature importance bar graph*, *Feature permutation importance*, *Learning curves*, *Precision-recall curves*, *Partial dependencies*, *Compute SHAP scores*, example decision trees. The ones marked *intensive* take a while |
| *Save model meta-data file* | keep the settings with the model |

* *Save settings only* stores the settings without training.
* *Pop out ⇱* opens the form in its own window.

The models are saved as `models/generated_models/<behaviour>.sav`, with the evaluations in
`models/generated_models/model_evaluations/`.

**Reading the evaluations:**

* The classification report's precision and recall on the held-out frames are the first check.
* The precision-recall curve shows the trade-off between them, which helps choose the threshold.
* Feature importance (and SHAP) show what the classifier relies on: features that make sense for
  the behaviour are a good sign.

## Validate

**Validate › Validate classifier** runs one model on one video it has not seen, and renders the result over the video to
check by eye:

| Option | What it does |
|---|---|
| Model path (.sav) · Feature file | the model, and the video's features |
| Discrimination threshold | the probability above which a frame counts as the behaviour |
| Minimum bout length (ms) | shorter bouts are dropped as noise |
| Show pose · animal names · classifier confidence · bounding box, font and circle sizes, palette | what the video shows |
| Gantt type | add a behaviour timeline beside the video |

Adjust the threshold and minimum bout here until the video looks right, then use the same values
for **Inference**.

## Inference

**Inference › Run inference** runs the classifiers on every video. Each behaviour has a row in the table: its model
(`.sav`, *Browse…*), its threshold (0–1) and its minimum bout length (ms). *Reload classifier list*
re-reads the behaviours. The settings are kept in `project.toml`.

The predictions go to `derived/classifications/<video>.parquet`: per frame, each behaviour's
probability and its 0/1 call. Analysis and Visualizations read them from there.

## Worked examples

[`../workflow_recipes.md`](../workflow_recipes.md) walks through:

* R1: one animal, one behaviour;
* R2: two animals' social behaviour, and identity swaps;
* R7: training from another program's annotations.
