# 12. Video processing, add-ons and tools

The last three pages hold utilities you reach for now and then.

* **Video Processing** edits videos.
* **Add-ons** has analyses for particular experiments.
* **Tools** converts files between formats.

## Video Processing

Each section works on one video or a whole folder (*Source*). Most can use the GPU encoder (NVENC)
when FFmpeg has it.

| Section | What it does |
|---|---|
| Format conversion | convert to MP4, AVI, MOV or WEBM, with a codec and quality; keep the audio |
| Trim & split | cut by frame numbers or times; *Multiple segments* cuts several clips at once (`start-end, start-end`) |
| Crop & mask | crop to a rectangle, circle or polygon you draw. *Multi-crop* makes several videos from one, one per drawn region (e.g. four cages filmed together) |
| Resize & rate | change the resolution (down or up, by size or scale factor) or the frame rate |
| Rotate & flip | rotate by a preset or any angle; flip horizontally or vertically |
| Filters & enhancement | CLAHE contrast, brightness and contrast, blur, greyscale, black-and-white threshold |
| Background removal | subtract a static background: from a reference image, a window of the video, or the whole video; with background and foreground colours and a threshold |
| Overlay / burn-in | burn in a frame count, timer, text, the file name, a watermark, a progress bar or another video |
| Frame extraction | save frames as images: a range, a format, greyscale or CLAHE, with the video's name in the image names |
| Merge frames → video | make a video from a folder of numbered images |
| Import frame directory | copy (or link) a folder of frames into the project |
| Join & transition | join videos one after another, side by side, stacked or as a mosaic; a crossfade between two |
| Image format conversion | convert images to PNG, JPEG, BMP, WEBP or TIFF (or one multi-page TIFF) |
| Utilities | reverse a video; change its speed; calibrate pixels per mm by clicking two points; check that seeking to every frame works (some files don't, which breaks frame-exact tools) |
| Metadata & audit | an average frame over a video or a stretch of it (a clean picture of the arena) |

**Tools menu.** The menu bar's **Tools** has the most used of these: *Reverse video…*, *Crossfade
two videos…*, *Print video metadata…*, *Check video seekable…*, *Calibrate pixels/mm…* and *Change
video playback speed…*.

## Add-ons

Analyses for particular experiments:

| Section | What it does |
|---|---|
| Cue-light — data analysis | from cue lights defined as ROIs: when each light is on, how often, for how long |
| Cue-light — classifier statistics | the behaviours before and after each light comes on (*Pre-* and *Post-activation window*) |
| Cue-light — movement statistics | the animal's movement around each light coming on |
| Cue-light — visualizer | the lights' states (and the pose) over the video |
| Kleinberg burst smoothing | merges classifier bouts that come in bursts and drops isolated ones, with Kleinberg's burst detection (*Sigma*, *Gamma*, *Hierarchy level*, or the best level found automatically) |
| Mutual exclusivity corrector | when two behaviours that can't happen together are both called, keep one, by rules in a CSV |
| Pup retrieval | scores when a dam retrieves pups into the nest ROI, from the dam's and the pups' body-parts and an approach distance |
| Spontaneous alternation | Y-, T- or radial-maze alternation from arm and centre ROIs: entries, alternations, the alternation rate (with a video, if asked) |
| Directing — body-part settings | the body-part used by *Directing toward body-part* in **Analysis** |
| Blob tracker — initialise · Blob quick-check | track an animal as a moving blob against the background (no keypoints): tune the detection, then check it |

The experiments are walked through in [`../workflow_recipes.md`](../workflow_recipes.md): R4 cue
lights, R5 spontaneous alternation, R6 pup retrieval.

## Tools

Utilities that need no project:

| Section | What it does |
|---|---|
| Convert pose / annotation data | between formats: DLC, multi-animal DLC (CSV and H5), SLEAP (`.slp`, CSV, H5), Labelme (keypoints, boxes, annotations), COCO keypoints, a SimBA / Mufasa project → YOLO keypoints or boxes, Labelme, DLC, a CSV table or images |
| Re-order pose keypoints | change the order of the keypoint columns in a folder of DLC or maDLC files (*Load order* reads the current order) |
| SLEAP → YOLO conversion | a YOLO keypoint training set from SLEAP predictions ([chapter 10](10-yolo-pose.md)) |
| SimBA ROIs → YOLO conversion | a YOLO detection training set from a project's ROIs (*OBB* for rotated boxes) |
| Export to CSV | a project's features, labels or both, as wide CSV files, for one video or all, for use in other software |
