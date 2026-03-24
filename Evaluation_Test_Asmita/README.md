# Printed Historical Text Recognition with CRNN
This folder contains a framework for carrying out OCR on Historical Spanish printed text using CRNN architecture and CTC loss.

I have implemented a three-stage OCR pipeline to address the problem statement in the Evaluation Test:
1. **DBNet** for text detection- to locate text regions within images.
2. **CRNN** for text recognition- converts the detected text regions into text.
3. **LLM-based refinement** to correct OCR errors and improve the quality of the final transcription while preserving historical spelling.

## Stage 1- Text Detection:
**Model Used:** Pre-Trained DBNet Model  

**Basic Idea :** DBNet approaches scene text detection as a pixel-wise segmentation (into text/
not text) problem. Given an input image, a convolutional backbone (e.g., ResNet) is used to
extract deep feature maps. A segmentation head is then used to predict a probability map
where each value corrosponding to a pixel represents the likelihood of the pixel belonging to a
text region.
Instead of using a fixed threshold (like more traditional approaches) to convert this
probability map into a binary mask, DBNet introduces the concept of a learnable threshold
map.This allows the model to adaptively determine decision boundaries at each pixel
location.

**Differentiable Binarization:** Traditional binarization applies a hard threshold:

$$
B(x,y) = \begin{array}{ll}
1 & \text{if } P(x,y) > T(x,y) \\
0 & \text{otherwise}
\end{array}
$$

However, this makes it non-differentiable and thus non-trainable.
The innovation of the DBNet Model is the use of the following approximation:

$$
\hat{B}(x,y) = \sigma\big(k(P(x,y) - T(x,y))\big)
$$

This approximates hard thresholding while remaining differentiable, thus enabling gradients to flow through both the probability and threshold maps during training. Thus, the complete model can be trained end-to-end.

**Reasons for Choosing DBNet:**
1. Works well with multi-oriented and curved text- In our use case, images of Historical documents may be non-horizontal/curved and handling these cases is crucial.
2. The model demonstrates strong performance on the standard scene text benchmarks.

**Reference** :  
Liao, M.; Wan, Z.; Yao, C.; Chen, K.; & Bai, X. (2019).  
*Real-time Scene Text Detection with Differentiable Binarization.*  
https://arxiv.org/abs/1911.08947

### Implementation Details :  
A pre-trained DBNet model from the DocTR library is used for scene text detection.

Given an input image, the model predicts text regions and outputs corresponding bounding boxes. These bounding boxes are then used to extract individual text blocks, where each block typically corresponds to a single line of text.The extracted line-level text images are subsequently passed to the OCR pipeline for recognition.

Below is a sample visualization of text regions detected using DBNet:

<p align="center">
  <img src="Model Architecture Images/Text_Detection_Sample.png" width="600">
</p>

Ground truth label data is prepared from the page- wise transcriptions provided. The final data is in the form of (Cropped Line Image, ground Truth Line Text) pairs.



## Stage 2- CRNN Modelling

**Dataset Preparation:**
The dataset is first shuffled, and then split into Train, Validation and Test Datasets.A data augmentation pipeline is implemented on the Train data to prevent the model from overfitting. The Character Set for the model is also defined.

### Theory and Reasoning Behind Models/Frameworks Used- The CRNN Model:

**Basic Idea:**

The CRNN Model was first introducrd in a paper by Shi, Bai & Yao in 2015.It is a combination of:
1. DeepCNN layers
2. Followed by RNN (Deep Bi-Directional LSTM) layers
3. CTC Transcription layer


### Motivation for CRNN:

**Why are traditional CNNs not suitable for this task?**

1. CNNs are effective for image classification tasks where the goal is to predict a single
label for an input image. However, image-based text recognition requires prediction of
**a sequence of labels** rather than a single label.

3. CNN architecture typically produces fixed output dimensions. In OCR tasks, the length
of the target output sequence can vary significantly.

Thus, standard CNN architecture alone is not well suited for text recognition tasks.

**Why are RNN-style models useful for this task?**

1. Recurrent Neural Networks are specifically designed to **model sequential data**. In
image-based text recognition, characters appear sequentially along the horizontal 
direction of the image which can be modelled with RNN.

3. Characters can often be recognized more reliably when their surrounding context is
considered. For example, ambiguous characters such as “i” and “l” can be distinguished
more easily when neighboring characters are taken into account. (**Reference** -Shi, Bai &
Yao (2015)- An End-to-End Trainable Neural Network for Image-based Sequence
Recognition and Its Application to Scene Text Recognition)

5. RNNs can **process sequences of variable length**, making them suitable for OCR tasks
where different images contain different numbers of characters.

To capture long-range dependencies in text sequences, CRNN uses **Long Short-Term
Memory (LSTM) units**, which solves the vanishing gradient problem present in traditional
RNNs.


## Model Architecture:

<p align="center">
  <img src="Model Architecture Images/CRNN_Model_Architecture.png" width="600">
</p>

### 1. Convolutional Layers:
   
* CNN layers **extract visual feature maps** from the input image.

* Workflow is as follows: image → convolutional feature maps → Map-to-Sequence transformation
→ feature sequence.

* Each **column in a feature map corresponds to a receptive field** in the original
image.Thus, the model effectively reads the image left to right.

<p align="center">
  <img src="Model Architecture Images/Receptive_Field.png" width="600">
</p>

* The feature maps are converted into a **sequence of feature vectors**- by concatenating
the corresponding columns of all feature maps together to form a feature vector. Thus,
this converts image data into sequential data.

* CRNN removes the fully connected layers used in traditional CNNs and uses only
convolution and pooling layers.

* All input images must be normalized to a fixed height.The width is scaled
proportionally to preserve the aspect ratio of the image.


### 2. Recurrent Layers:

* The feature sequence generated from the feature maps are fed into **deep bidirectional
LSTM layers**.

<p align="center">
  <img src="Model Architecture Images/Bidirectional_LSTM.png" width="600">
</p>

* Bidirectional LSTM is used to capture context from both directions, improving
recognition accuracy. 

* Multiple bidirectional LSTM layers are stacked to enable the
network to learn higher-level sequential representations.


### 3. Transcription Layer:

* The transcription layer converts the sequence of predictions produced by the LSTM
network into the final text sequence.

* CRNN uses **Connectionist Temporal Classification (CTC)** to compute the probability
of label sequences.CTC introduces a special "blank" token and allows repeated labels in
intermediate predictions. A mapping function collapses repeated labels and removes
blanks to produce the final output sequence.The final prediction is obtained by
**selecting the label sequence with the highest probability.**

* NOTE: CRNN **does not require character-level alignment** between the input image and
the output text.Older OCR models needed character-level labelled datasets.CRNN
instead performs sequence recognition directly- this makes training dataset peperation
easier.


### Model Training:

The model is trained by **minimizing the negative log-likelihood of the predicted label
sequence, conditional on the input image.**

Key **Advantages**:

1. CTC enables the model to learn from image–text pairs directly, without requiring explicit
segmentation of characters- thus **character level annotations are not required**.

2. Entire architecture can be **optimized jointly** in an end-to-end manner.
   
3. The model can handle variable length sequences naturally.

**Reference**: Shi, Bai & Yao (2015)
An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its
Application to Scene Text Recognition
https://arxiv.org/abs/1507.05717


### Reasoning behind using pre-trained weights:

The prepared dataset contains a small number of training samples (< 800 line images), which
is typically insufficient to train a deep neural network like CRNN from scratch. Therefore,
instead of initializing the model with random weights, we start with pre-trained weights
obtained from models trained on large synthetic OCR datasets. These pretrained models
have already learned general visual features of text such as character strokes, edges, and
spatial patterns.

By **fine-tuning the pretrained model** on the available dataset, the network can adapt these
learned representations to the specific characteristics of the task at hand. This **Transfer
Learning** approach significantly improves training efficiency, reduces overfitting, and enables
good recognition performance even with a relatively small dataset.


### Implementation Details:

The **Deep Text Recognition Benchmark** framework provides several downloadable pretrained models that can be used as a starting point to fine-tune OCR systems. These models have been trained on large synthetic text datasets such as MJSynth (Synth90k) and
SynthText, which contain millions of generated word images rendered using a wide
variety of fonts, backgrounds, and distortions.

A **Four-stage Scene Text Recognition framework** was introduced in the benchmark paper :
Baek et.al (2019)
What Is Wrong With Scene Text Recognition Model Comparisons? Dataset and Model Analysis
https://arxiv.org/pdf/1904.01906

The framework the paper described (and the Structure followed by the Deep Text Recognition Framework) is as follows:
Transformation → FeatureExtraction → SequenceModeling → Prediction

**What is the Transformation stage?**

Transforming the input image X into a normalized image X˜.This stage typically uses a TPS Network, which 
normalizes the input text image by correcting geometric distortions such as slight
curvature, perspective skew, or irregular alignment. 

In our case, the dataset consists of
cropped line images from scanned documents where the text is largely horizontal and
well aligned. Therefore, a TPS transformation is not expected to significantly affect
recognition performance.

Additionally, a TPS layer is not present in the CRNN architecture intoduced in the Shi, Bai &
Yao (2015) paper.

For these two reasons, I am **not including a Transformation layer in my model architecture**.


**Final Model**- I will be implementing the following framework : **None-ResNet-BiLSTM-CTC**
(following Shi, Bai & Yao (2015) paper)

Pre-trained model weights download link: https://drive.google.com/drive/folders/15WPsuPJDCzhp2SvYZLRj8mAlT3zmoAMW

## Model Training:


### Model Training Stage 1- Freezing all CNN Parameters and training Bi-LSTM and Prediction Layer:

In the first stage of model training, all parameters of the feature extraction layer (CNN)
are frozen. During this phase, only the weights of the Bi-LSTM and the prediction layer are
updated. The **initial layers of CNNs** typically learn **general visual features** such as edges,
curves, and basic shapes, which are largely **transferable** across datasets. Therefore, freezing
the CNN helps preserve the useful representations learned during pretraining.

Training the entire model simultaneously from the beginning can lead to **large weight
updates** that may overwrite these useful features and increase the risk of overfitting,
particularly when the training dataset is small. By freezing the CNN layers initially, the model
retains the previously learned feature representations while allowing the **sequence modeling
(Bi-LSTM) and prediction layers to adapt to the characteristics of the current dataset**.

The **final prediction layer is replaced** with a new layer with randomly initialized weights and adjusted number of output classes [len(Character Set) + 1]. The Model is trained for 10 Epochs.

## Defining Model Evaluation Metrics- CER and WER :

### Character Error Rate (CER):
Measures how many characters in the predicted text are wrong compared to the ground
truth. It is calculated using **Levenshtein distance**, which counts the minimum number of
changes needed to convert one string into another. 

The allowed edits are:
1. Insertion (I) : adding an extra character
2. Deletion (D) : missing a character
3. Substitution (S) : replacing one character with another

**Formula:**

$$
\text{CER} = \frac{S + D + I}{N}
$$

Where:
S = Number of substitutions  
D = Number of deletions  
I = Number of insertions  
N = Number of characters in the ground truth  


### Word Error Rate (WER):
Measures how many words in the predicted text are wrong compared to the ground truth.
Uses the same formuala as CER, just words are the units of measurement as opposed to
individual characters.


### Why are we evaluating model using CER/WER and not the CTC loss value?

The model **is trained by minimizing the CTC loss** -i.e, the parameters are updated using this
loss in the training process. A lower CTC loss also means that the model assigns higher
probability to the correct sequence.

However, this probability is calculated **before decoding** and thus includes many possible
alignments.CTC uses probabilities of all valid alignments, not just the final decoded text. The
final output also **depends on the decoding method used**, not only the predicted alignments.

CER evaluates the quality of the final predictions after decoding, which depends on both the model outputs and the decoding strategy. This metric is important because, in OCR systems, what ultimately matters is
the **accuracy of the final decoded prediction.**

Thus, in the model training, at each step, the model which gives the **lowest CER on the Validation
dataset** (and not the model with the lowest Validation loss) has been picked up for the
subsequent steps.

### Model Training Stage 2 (Training Full Model):
In the second stage of training, the entire model is unfrozen and trained end-to-end using a
**lower learning rate**. This allows the network to fine-tune all parameters while minimizing the
risk of overwriting the useful features learned during earlier training stages.

## Full Model Training- Evaluation Metrics on Validation Data:
The full model was trained for **20 Epochs** and the best Character Error Rate achieved on
**Validation Data** was **9.63%** .

Thus, the model **predicted about 90.4% of the characters from the Validation dataset
correctly and 9.6% of the characters incorrectly.**


## Output refining using LLM Integration:

**The Goal**: Use a trained large language model to refine the predictions of the CRNN model.

**Bonus Idea**: Providing a few pairs of ground truth labels and OCR predictions from the
train dataset to a Large Language Model. Prompt the LLM to analyze the patterns and differrences between
the predictions and the ground truth and remember this information. Next, input the OCR
predictions, one line at a time, to the prompt and ask the LLM Model to predict the ground
truth from this predicted OCR output. Try differrent prompting patterns and select the
one with the lowest CER as the final prompt. Evaluate this best prompt on the test data to get
the final accuracy score.

This is method is known as **'Few Shot Prompt Tuning'**

**Implementation Details:**

To integrate an LLM-based post-processing step into the OCR pipeline, local language
models were deployed using the **Ollama framework**. Relying on cloud APIs such as Google
Gemini can introduce rate limits and usage restrictions, which may hinder large-scale batch
processing of OCR outputs. To avoid these constraints and ensure uninterrupted processing,
Ollama was used to run open-weight language models directly on local hardware, enabling
efficient and unrestricted local inference.

The models I have experimemted with are as follows:
1. Mistral 7B
2. Qwen 2.5 7B

The best CER Error rate that was achieved on Passing OCR predictions of Training Data
through a Large Language Model was 13.2%. 

Thus, in the current framework, applying LLM-based post-processing to the CRNN outputs did 
not improve overall performance. This suggests that the current LLM refinement approach
implemented requires further redesign and optimization.

## Model Evaluation Results on Test Dataset:
On running the CRNN model on the **Test Dataset, about 87% of characters were
predicted correctly. Thus, 13% of characters in the predicted text differed from the ground
truth label.**

On the **Validation dataset, 90.4% of the characters were predicted correctly and 9.6% of
the characters were predicted incorrectly.**


## Conclusion:
An end-to-end OCR pipeline for historical Spanish printed text has been implemented using
DBNet for text detection and CRNN for text recognition. The trained model achieved a CER
of 13% on Test data and 9.6% on Validation data demonstrating the effectiveness of the
implemented framework.

Preliminary experiments with LLM-based post-processing were attempted. However, this did
not result in significant improvements in CER within the current framework, indicating that
the LLM- Based refinement strategy requires further redesign.

Future work will focus on the following directions:
1. Implementing beam search decoding in place of greedy decoding to improve sequence
prediction.
2. Developing a more robust LLM-based post-processing framework for refining CRNN
outputs.
3. Exploring alternative text recognition architectures beyond CRNN to further improve
OCR accuracy.

