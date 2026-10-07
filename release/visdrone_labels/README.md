# VisDrone day / night hand labels

Hand labels and manifests made for Dawn's earlier regime (synthetic night from VisDrone-DET
daytime frames, enhanced back to day). Dawn no longer trains on VisDrone: the project moved to
real day thermal / RGB pairs from DroneVehicle. Nothing in the `dawn` package reads these files.
They are kept here for a separate release of their own, which is **pending**.

## What is here

| path | what it is |
|---|---|
| `visdrone_labels.csv` | One row per VisDrone-DET video sequence (or per single image), labelled by hand as `day`, `night`, `dusk` or `exclude`. Columns: `key`, `label`, `note`, then helper columns (`sheet_row`, `split`, `frames`, `log10_lum`, `noise`, `highlight_ratio`) from the labelling sheets. |
| `manifests/<split>/{day,dusk,night}.txt` | Image IDs per label, for the three VisDrone splits `train`, `val` and `test-dev`. Excluded and unlabelled images are in none of them. |
| `manifests/{train_day,val_day,test_day,hard_test,night_normal}.txt` | The five training manifests of the old regime, one image ID per line. |

## How the labels work

- A `key` is a sequence ID (the first field of an image ID, e.g. `0000287`), which labels every
  frame of that video, or a full image ID, which overrides its sequence.
- `dusk` and `exclude` frames were neither a clean day source nor a real night test, so the old
  pipeline dropped them.
- A **group** is a video sequence ID, or the collection ID for a `99999xx` still collection.
  Consecutive frames of a VisDrone video are near-duplicates, so the split into
  `train_day` / `val_day` was made by whole groups (10% of groups held out as `val_day`, seed
  `20261003`), and `test_day` is the daytime part of `test-dev`, left as the dataset ships it.

## The two night manifests

Frames labelled `night` were split by median linear luminance of the 256-px linearized image:

- `hard_test.txt`: median luminance <= **0.005**. 20 frames that are genuinely near black.
- `night_normal.txt`: median luminance > 0.005. 100 ordinary usable night frames (the lowest is
  about 0.0086, so there is a clean gap around the cutoff). For qualitative comparison only.

## The `99999xx` still collections

The 49 `99999xx` collections (6,497 stills, day and night interleaved) carry no hand label. They
were accepted as day automatically: a still counts as day when its median luminance is high
enough and its highlight ratio (99.5th percentile over median luminance) low enough. Both cutoffs
were fitted on the hand-labelled video frames to make **zero** errors there, then shrunk by 10% of
their span away from the nearest non-day frame. A 200-image contact sheet of what the cutoffs
accepted was written for spot-checking by eye. Those images are in `train_day`, `val_day` or
`test_day` only through that rule, not through a hand label.
