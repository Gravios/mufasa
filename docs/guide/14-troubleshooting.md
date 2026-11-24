# 14. Troubleshooting

Start with `mufasa doctor`: it shows what is installed and what isn't, and which project Mufasa
sees. Include its output when asking for help.

## Installing and starting

**`mufasa` reports something missing on start.** The workbench checks OpenCV, FFmpeg and the
display first and lists what's missing:

* the Qt front end needs the `qt` extra (`pip install -e ".[qt]"`);
* a compile error on install means a C compiler is needed for the native kernels.

**The window doesn't open over SSH.** The workbench needs a display. Use the command line instead
([chapter 13](13-command-line.md)), or forward X (`ssh -X`).

**`mufasa help` opens the workbench.** An older Mufasa, from before these commands, is first on
the `PATH`, often from another conda environment. `which mufasa` shows which one runs.

## GPUs

**`torch … no CUDA` in `mufasa doctor` on a machine with a GPU.** PyTorch was installed without
CUDA, or for a newer CUDA than the driver supports.

* PyTorch 2.11 and later from PyPI are CUDA 13 builds. With an older driver, install from
  `https://download.pytorch.org/whl/cu128`.
* Blackwell GPUs (RTX 50xx, RTX PRO 6000) need PyTorch 2.7 or later.

`mufasa-pose3d gpu-check` runs a synthetic 3-D solve on the CPU and the GPU and says what is
missing.

**Videos are slow to write.** Without NVENC, videos are encoded on the CPU. `mufasa doctor` shows
whether FFmpeg has NVENC; the status bar shows `NVENC ✓` when it does.

## Projects and files

**A project opened over the network shows no videos, or everything is empty.** The network mount
is probably stale. `ls` the project folder: if that hangs or is empty, remount it, then reopen the
project. Imported videos are relative links, so they work from any mount point once it's
reachable.

**Videos are missing.** Only `sources/videos/` (and its subfolders) holds a project's videos.
Import videos elsewhere with **Data Import › Import video**. `mufasa list videos` shows what the
project has.

**Imported pose gives empty features.** The pose file's markers don't match the project's. Check
with **Data Import › Import pose data** › *Add sessions* › *Check*, or look at the file in **Data inspector**. After
retraining the tracker with new markers, use **Re-import skeleton model**
([chapter 3](03-importing-data.md)).

**Distances are in pixels, or wildly off.** The videos aren't calibrated, or the calibration is
wrong. Check **Preprocessing › Video Calibration** ([chapter 4](04-preprocessing.md#video-calibration)).

**Results from an earlier run are used.** Each step reads the newest run of the step before it.
`mufasa list runs` shows the runs; delete an unwanted one's folder, or choose the input
explicitly in the form.

## 3-D

**3-D heights look off.**

* Check the camera: `mufasa list calibrations`. A wrong camera height or box size scales every
  height.
* Cages wider at the top need *Walls flare outward*.
* A head placed far too high usually means the ear labels came too close together (the head was
  turned, or seen end-on).

**The 3-D viewer doesn't find the video.** The video's name must begin the 3-D file's name
(`K7_cam.mp4` for `K7_camDLC_…_3d.parquet`). `mufasa info X_3d.parquet` shows which video it
finds. In the viewer, **Video…** picks one by hand.

**The 3-D viewer is wider than the screen.** Its toolbars wrap onto more rows when the window is
narrowed, so drag the window's edge in.

## Classifiers

**A classifier calls the behaviour everywhere, or nowhere.** Adjust the *Discrimination threshold*
and *Minimum bout length* in **Validate**, by eye, then use those values in **Inference**. If
precision or recall in the classification report is low, label more frames, especially where the
classifier is wrong (**Annotation › Review classifier predictions**).

**Training is slow.** Lower *RF estimators*, and leave the evaluations marked *intensive* (SHAP,
learning curves, permutation importance, partial dependencies) off until the final model.

## Reporting a problem

Include:

* the output of `mufasa doctor`;
* the steps that led to the problem, page and section;
* the error message. The terminal you started `mufasa` from shows the full traceback.
