# 10. YOLO pose

The **YOLO pose** page trains your own pose tracker, a YOLO keypoint model (Ultralytics), and runs
it on videos. This is an alternative to DeepLabCut or SLEAP. It needs a CUDA GPU and the
`ultralytics` package (`pip install ultralytics`); the page says so in red when either is
missing.

```text
labelled frames (SLEAP, DLC, Labelme, COCO, a Mufasa project)
        │  Tools › Convert pose / annotation data   or   Tools › SLEAP → YOLO conversion
        ▼
 a YOLO dataset (images, labels, map.yaml) ─► train ─► weights (.pt) ─► inference ─► pose files
```

## Make a training set

YOLO trains from a folder of images and keypoint labels, described by a `.yaml` map. Make one with
the **Tools** page ([chapter 12](12-video-addons-tools.md#tools)):

* **Convert pose / annotation data**: from DLC, multi-animal DLC, SLEAP (`.slp`, CSV, H5), Labelme,
  COCO keypoints, or a SimBA / Mufasa project, to *YOLO keypoints*;
* **SLEAP → YOLO conversion**: SLEAP predictions with options. *Frames per video* and *Train size*
  set the sample and the split. *Detection threshold* drops unsure frames, and *Padding* pads the
  boxes. *Greyscale output*, *Apply CLAHE* and *Single-instance* (collapse animals into one) are
  further options.

## Train

**YOLO pose › YOLO pose — train**:

| Option | What it does |
|---|---|
| YOLO map (.yaml) | the dataset's map |
| Initial weights (optional, .pt) | start from a pretrained model (faster and better than from scratch) |
| Save directory | where the run goes |
| Device · CPU workers | the GPU to use; data-loading workers |
| Epochs · Patience | how long to train; stop early after this many epochs without improving |
| Image size · Batch size | the training image size (256–1280; larger sees finer detail, uses more memory) and the batch |
| Export format | also export the trained model (e.g. ONNX, TensorRT) |
| *Generate plots* · *Verbose progress logging* | training curves; more output |

Training takes hours, so it runs in a separate process: the workbench doesn't wait, and a message
says where to watch its progress.

## Run

**YOLO pose › YOLO pose — inference**:

| Option | What it does |
|---|---|
| Model weights (.pt) | the trained model |
| Mode · Input | one video, or a folder of videos (*Recursive video search* for subfolders) |
| Device · CPU workers · Batch size · Image size | how it runs |
| Box threshold · IoU | how sure a detection must be, and how much two detections may overlap before one is dropped |
| Output format | the pose files' format |
| Max tracks · Max tracks per ID | how many animals to track |
| Smoothing (ms) · *Interpolate gaps* | smooth the keypoints; fill short gaps |
| *Stream (lower memory use)* | read the video as it goes, for long videos |

The output is pose files that **Data Import** reads as *YOLO-pose* ([chapter 3](03-importing-data.md)).
To check them by eye, use **Visualizations** › *YOLO pose predictions (external)*.
