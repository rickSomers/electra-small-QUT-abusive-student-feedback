---
language: en
license: apache-2.0
datasets: '-QUT-SVS-2021-2022'
base_model:
- google/electra-small-discriminator
tags:
- QUT
- University
- SET
- Student Evaluation of Teaching
- Harm
- Safety
- Wellbeing
pipeline_tag: text-classification
library_name: transformers
metrics:
- accuracy
- precision
- recall
- f1
- roc_auc
widget:
  - text: "The pacing was a little slow and boring"
    output:
      - label: Acceptable
        score: 0.996
      - label: Unacceptable
        score: 0.004
  - text: "He clearly favoured the female students"
    output:
      - label: Acceptable
        score: 0.014
      - label: Unacceptable
        score: 0.986
  - text: "They are too old to be a good teacher"
    output:
      - label: Acceptable
        score: 0.028
      - label: Unacceptable
        score: 0.927
  - text: "I learnt so much, thank you for the great semester!"
    output:
      - label: Acceptable
        score: 0.996
      - label: Unacceptable
        score: 0.004
---

# ELECTRA small discriminator QUT abusive student feedback
## Table of Contents
- [Model Details](#model-details)
- [How to Get Started With the Model](#how-to-get-started-with-the-model)
- [Uses](#uses)
- [Risks, Limitations and Biases](#risks-limitations-and-biases)
- [Training](#training)
- [Evaluation & Updates](#evaluation-and-updates)

## Model Details
**Model Description:** This is a fine-tuned checkpoint of [electra-small-discriminator](https://huggingface.co/google/electra-small-discriminator), fine-tuned on Queensland University of Technology ([QUT](https://www.qut.edu.au/)) Student Voice Survey (SVS) comments.</br>

The model is used to automatically detect unacceptable comments left by students in QUT's Student Evaluation of Teaching (SET) surveys, the SVS. 

Examples of unacceptable comments are those containing abuse, insults, discrimination, indications of a risk of harm, or otherwise deemed unacceptable according to the QUT [Evaluation of Courses, Units, Teaching and Student Experience Policy](https://mopp.qut.edu.au/document/view.php?id=144).

Comments are labelled as Acceptable (0) or Unacceptable (1), with the probability of the classification used to a further discern which comments are screened manually by University staff.

This model serves as part of QUT's methodology for screening SVS comments to prevent harm to the wellbeing and career prospects of academics, further detailed [here](https://eprints.qut.edu.au/233735/1/112912754.pdf).

- **Developed by:** 
  - Rick Somers - Evaluation Support Officer, QUT
  - Sam Cunningham - Senior Lecturer, QUT
- **Model Type:** Text Classification
- **Language(s):** English
- **License:** Apache-2.0
- **Parent Model:** For more details about ELECTRA, we encourage users to check out [this model card](https://huggingface.co/google/electra-small-discriminator)
- **Resources for more information:**
    - [Model Documentation](https://huggingface.co/docs/transformers/main/en/model_doc/electra#transformers.ElectraForSequenceClassification)
    - [ELECTRA paper](https://arxiv.org/abs/2003.10555)

## How to Get Started With the Model

Example of single-label classification:

```python
import torch
from transformers import AutoTokenizer, ElectraForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained("rickSomers/electra-small-QUT-abusive-student-feedback")
model = ElectraForSequenceClassification.from_pretrained("rickSomers/electra-small-QUT-abusive-student-feedback")
inputs = tokenizer("Thank you for your engaging tutorials", return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
predicted_class_id = logits.argmax().item()
model.config.id2label[predicted_class_id]
```

## Uses

### Direct Use

This model is intended be used for text classification. You can use this finetuned model directly for acceptable/unacceptable text classification, or it can be further fine-tune for downstream tasks.

### Misuse and Out-of-scope Use

This model should not be used without supporting processes and/or human evaluation. It should not be used to intentionally create hostile, harmful, or alienating environments for people.

## Risks, Limitations and Biases

Based on the nature of the training data being from a large, urban, Australian university, we observe that this model could produce biased predictions.

The performance of the model is therefore unknown in other contexts. While we believe that it could be beneficial for other instutes to adopt it in their SET screening, nuisanced differences in question type, language styles and choices, and local context could lead to varying performance.

We strongly advise users to thoroughly investigate their needs and use-case for this model, and evaluate its performance thoroughly before formally implementing into policies and processes. It is not recommended to use this model as the sole screening mechanism for harmful content.


# Training

### Training Data

The model was trained on 2021-2022 QUT SVS qualitative survey responses. Each comment was asigned a label of 0 or 1, acceptable or unacceptable. Labels were asigned from previous keyword matching algorithms, deprecated unacceptable machine learning models, and staff raised lists.

### Training Procedure

Training was undertaken over several months to produce optimal performance. Oversampling techniques were used routinely to better balance the dataset due to a relatively small number of unacceptable comments.

#### Fine-tuning hyper-parameters

- learning_rate = 5e-5
- batch_size = 8
- max_position_embeddings = 512
- num_train_epochs = 2


# Evaluation and Updates

The performance of the model on new SVS responses are continually monitored and improved. 

Internally, the model is routinely evaluated on the each semesters' data. To improve performance, it is often completely retrained from its base model again, or simply further fine-tuned with the inclusion of new text data.

Note that the model uploaded here is a deprecated version (trained on only 2021 & 2022 data) of the current model used by QUT.