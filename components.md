## BERT Based Parser

Stored in the folder `bertclassifier`. Defined in the file `parser.py`.

### Downloads Required

The data files for both casino and emotion dataset are already present in the repo, if the training needs to be repeated. The trained model files can be download from the shared onedrive folder. After downloading the model files, place them in the folder as `bertclassifier/models/emotion_classifier.pt` and `bertclassifier/models/intent_classifier.pt`.

### Installation Required

Major libraries required are `torch` and `transformers`. No other unique library required.

### Meta-Data

For emotion classification, we predict any one of the following 6 classes:

```
0: "anger"
1: "fear"
2: "joy"
3: "love"
4: "sadness"
5: "surprise"
```

The accuracy achieved on testing/validation dataset was as follows,

```
Validation Accuracy : 0.9255
Validation MCC Accuracy : 0.9033772608381565
```

For intent classification, we predict the following 10 classes (multi-label classification):

```
0: "elicit-pref"
1: "no-need"
2: "uv-part"
3: "other-need"
4: "showing-empathy"
5: "vouch-fair"
6: "small-talk"
7: "self-need"
8: "promote-coordination"
9: "non-strategic"
```

The accuracy achieved on testing/validation dataset was as follows,

```
Accuracy Score : 0.6103896103896104
F1 Score (Micro) : 0.7356115107913669
F1 Score (Macro) : 0.528857752869762
```

### Train and Test Model

The training/testing code is combined together in `bertclassifier/train.py` for both the classifiiers and can be run as follows,

```
python train.py emotion
python train.py casino
```

## Seq2Seq Model for Baseline Negotiation Agent

### Downloads Required

### Installation Required

### Meta-Data

### Train and Test Model

## Dialogue Generator

### Downloads Required

### Installation Required

### Meta-Data

### Train and Test Model

## RL Agent

### Downloads Required

### Installation Required

### Meta-Data

### Train and Test Model
