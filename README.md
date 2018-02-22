# Progressive Virtual Image Generation using UnrealCV

This repository presents source code necessary to reproduce the automation flow addressed in the following paper:
@unpublished{Choi2018ISCAS_ProgressiveImgGen,
  title={Stochastic Functional Verification of DNN Design through Progressive Virtual Dataset Generation},
  author={Jinhang Choi and Kevin M. Irick and Justin Hardin and Weichao Qiu and Alan Yuille and Jack Sampson and Vijaykrishnan Narayanan},
  note={Accepted to International Symposium on Circuits and Systems (ISCAS)},
  year={2018}
}

# Prerequisites
To run scripts included in this repository, you have to prepare for
- [Unreal Engine 4](https://github.com/EpicGames/UnrealEngine) with [UnrealCV](https://github.com/unrealcv/unrealcv)
- [xmljson](https://pypi.python.org/pypi/xmljson) Python package for json_parser/tools/xml2json.py
- [Torch7](http://torch.ch/) with [Coco](https://github.com/cocodataset/cocoapi)
- NVIDIA [DIGITS](https://github.com/NVIDIA/DIGITS) for visualizing object classification
- Facebook [SharpMask](https://github.com/facebookresearch/deepmask) for retreiving object localizations