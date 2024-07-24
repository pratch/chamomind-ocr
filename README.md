# chamomind-ocr

## Install requirements
- python -m venv chamomind-ocr
- (Windows) chamomind-ocr\Scripts\activate
- (Linux) source chamomind-ocr/bin/activate
- pip install -r requirements.txt

## Run end-to-end document field extraction
- python main.py \<image path\>

## Run experiments
- python -m experiments.create_dataset_from_ocr

## Install and run AnyLabeling
- pip install anylabeling
- anylabeling