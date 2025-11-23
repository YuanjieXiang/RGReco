# RGReco

This is the repository for RGReco, A Unified Framework for Automated R-Group Recognition in Chemical Literature Images!

<img width="3197" height="1998" alt="image" src="https://github.com/user-attachments/assets/82eba44d-6a81-488b-800b-f6669fa92253" />


## Quick Start

### Installation

**Step1: Prerequisites**

- **Python** >= 3.10
- **Java Runtime Environment** (JRE) 8 or higher

```
conda create -n rgreco python=3.10 # python >= 3.10
conda activate rgreco
# For GPU
conda install pytorch=2.5.1 torchvision torchtext pytoch-cuda=12.1 -c pytorch -c nvidia  
pip install onnxruntime-gpu>=1.21
# For CPU only
conda install pytorch=2.5.1 torchvision torchtext -c pytorch  # cpu
pip install onnxruntime>=1.21
```

**Step 2: Clone the repository**

```
git clone https://github.com/YuanjieXiang/RGReco.git
cd RGReco
```

**step3: install the package and its dependencies**

```
pip install -r requirements.txt
```

### Example

First, modify the `src.settings` file

1. Download the model weights and modify the weight path in the settings file
2. The mode specifies the run mode, options are: cli, ui, and test:

```shell
# CLI mode requires an image_path argument for the input image (local path), default is test.png, e.g.:
python app.py --image_path test.png
# UI mode opens a screenshot interface, using the screenshot as input
# Test mode attempts to run all images in TEST_DIR
python app.py
```

### Model

Model weights and tools can be downloaded from: 

| Task                                                   | Description                                                  |
| ------------------------------------------------------ | ------------------------------------------------------------ |
| Molecule Segmentation                                  | Using ONNX format for inference to minimize dependencies. Original model from: [DECIMER-Segmentation](https://github.com/Kohulan/DECIMER-Image-Segmentation), Downloaded from [RGReco_Det](https://huggingface.co/yuanjier/RGreco_Det) |
| Attachment Point and  Substituent Identifier Detection | Downloaded from [RGReco_Det](https://huggingface.co/yuanjier/RGreco_Det) |
| Optical Chemical  Structure Recognition                | Downloaded from [MolScribe](https://github.com/thomas0809/MolScribe) |
| Document Content Recognition                           | Downloaded from [Surya](https://github.com/datalab-to/surya) |
| IUPAC Names  Recognition                               | Downloaded from [OPSIN](https://github.com/dan2097/opsin)    |

