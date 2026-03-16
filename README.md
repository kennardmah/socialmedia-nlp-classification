# This README file is a condensed version of the submitted essay [uploaded here on github](https://github.com/kennardmah/socialmedia-nlp-classification/blob/main/ADL___Paper__Classification_.pdf)

This paper was an optional project during my course at Columbia: COMSW4995 Applied Deep Learning taught by Prof. Andrei Simion.  It was fun way to apply what we were learning in class: NLP-based models ranging from NNs to Transformers.
I used a kaggle dataset on cybersecurity classification from text to apply RNN-LSTM, BERT, and RoBERTa to compare their initial results.

# Implementation and Comparative Analysis of Social Media Sentiment Classification: LSTM RNN vs. Fine-Tuned Transformer Models (BERT & RoBERTa)

Kennard Mah — Data Science Institute, Columbia University – ksm2198@columbia.edu

## Abstract

This paper addresses the increasing prevalence of cyberbullying in social media and proposes a method to detect harmful content through sentiment analysis and classification. With a dataset of over 47,000 social media threads, the study explores various methods, including RNN (LSTM), BERT, and RoBERTa, to classify cyberbullying. The RNN (LSTM) model with Word2Vec achieved an accuracy of 81.93%, showing effective performance in identifying cyberbullying but indicating room for improvement in distinguishing between different types of cyberbullying. BERT and RoBERTa models, which leveraged pre-trained transformers, outperformed the RNN (LSTM) model, with BERT achieving an accuracy of 85.78% and RoBERTa achieving 85.25%. Both transformer models demonstrated higher classification accuracy, though RoBERTa exhibited a more balanced precision and recall across categories.

Although all models showed promising capabilities for cyberbullying detection, transformer models, especially BERT and RoBERTa, offered superior performance in accurately classifying social media posts. BERT showed the highest accuracy, with minor Type I errors in misclassifying non-bullying threads as bullying. RoBERTa, while computationally more expensive, offered slightly better balanced performance across various metrics. These findings suggest that both RNN LSTM and pre-trained transformer models, BERT and RoBERTa, offer a robust framework for detecting cyberbullying in real-world social media data. For future research, further refinement in pre-processing of imbalanced categories and implementation of Adapters and LoRA could potentially improve performance.

**Keywords:** NLP Classification · Applied Deep Learning · RNN LSTM · Transformers · BERT · RoBERTa

## 1. Introduction

Cyberbullying has become more prevalent with the increasing use of social media across all age groups. A vast majority of people use social media for daily communication, and this ubiquity means that cyberbullying in this form can affect anyone. Given the ability to hide behind a screen, these personal attacks are much more undetectable, inconsequential, frequent, and unavoidable than traditional bullying. While it is difficult to protect people from the internet, there is a wealth of data available, especially in social media speech and tweets. By understanding the different types of discrimination—gender, religion, age, ethnicity, and others—we can teach our model to categorize and flag these threads to build a system that can combat the consequences of discriminatory speech.

<p align="center">
  <img src="figures/figure1.png" width="400"/>
  <br>
  <em>Figure 1: Sentiment Distribution of Dataset</em>
</p>

As displayed in the class distribution in Figure 1, this paper uses a dataset containing more than 47,000 social media threads (~7,800 per classification), each labeled with its category for cyberbullying. The dataset has a balanced representation, with duplicates removed and no null values.

By understanding the sentiment of social media posts and classifying them accordingly, we can create a data-driven method to better understand how we can combat these issues. Given the text-driven nature of cyberbullying, we can leverage this data to design models that detect harmful content more effectively.

### Paper Structure

- **Literature Review:** Examine existing techniques and tools for sentiment analysis, laying the groundwork for this study.
- **Methodology:** Propose a novel deep learning approach leveraging pre-trained transformer models to classify cyberbullying.
- **Results and Discussion:** Present the findings from the model's performance, analyzing its effectiveness in optimizing classification.
- **Conclusion:** Reflect on the real-world implications of the proposed method and explore opportunities for further research and model improvement.

## 2. Literature Review

This section is structured to better understand current methods used for NLP sentiment analysis, mainly in RNN-LSTM and fine-tuning transformer-based models, BERT and RoBERTa, and how they are evaluated.

### 2.1 Sentiment Analysis using NLP

Sentiment analysis is an emerging field in text mining. It is an NLP application that identifies a text corpus's emotional or sentimental tone or opinion—usually emotions or attitudes toward a topic in varying classes, essentially identifying and categorizing opinions expressed in a piece of text over different social media platforms [1].

**Recurrent Neural Networks (RNN)** are a form of ANN that can memorize length sequences of input patterns by capturing connections between sequential data types [2]. To avoid exploding and vanishing gradients, **Long Short Term Memory (LSTM)** networks are a type of RNN designed to mitigate these issues through gating mechanisms [3]. While RNNs are limited to looking back in time for approximately ten timesteps [4], LSTM networks are capable of learning more than 1,000 timesteps, depending on the complexity of the network [5]. This makes LSTM-RNNs a very powerful dynamic classifier for sequential data [6]. Its implementation with NLP has been seen in studies surrounding natural language inference [7] and sentiment analysis tasks by integrating Word2Vec for hotel reviews [8].

**Word2Vec** is a model for representing words in vector space, measuring the quality of these representations in word similarity tasks to assess syntactic and semantic word similarities. There are two models that allow transforming a word into a numerical vector: Continuous Bag-of-Words (CBOW) and Skip-Gram. CBOW predicts a center word using the words around it, while Skip-Gram predicts words around a center word [9]. Using a pre-trained Word2Vec embedding method, it is possible to integrate Word2Vec into an RNN-LSTM task [10], as used in this paper.

**Bidirectional Encoder Representations from Transformers (BERT)** is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. This enables BERT to be pre-trained and fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks [11]. Some examples in recent research include classifying mobile app star ratings based on user reviews [12] and sentiment analysis for Vietnamese reviews, outperforming other models that use GloVe and FastText for word embeddings.

**The Robustly Optimized BERT Approach (RoBERTa)** is a BERT variant introduced by Facebook AI Research as an improvement over BERT, removing the Next Sentence Prediction objective and using only the Masked Language Modeling objective. RoBERTa uses larger mini-batches and learning rates to learn more effectively with increased training data, while also employing dynamic masking to introduce variability and help models learn more flexible patterns [13].

Other methods explored outside of this paper to improve the performance of pretrained models are **Adapter** and **Low-Rank Adaptation (LoRA)**. An adapter is a small module inserted into pre-trained models like BERT to enable task-specific fine-tuning without significantly modifying the original model's weights, commonly used in transfer learning [14]. LoRA tackles the challenge of full fine-tuning with larger models by freezing the pre-trained model weights and injecting trainable rank decomposition matrices into each layer of the Transformer architecture, reducing the number of trainable parameters for downstream tasks [15].

### 2.2 Evaluation Metrics for Classification Models

Evaluation of classification models differs based on the input dataset and the task at hand, whether it is balanced or imbalanced. Upon research of 16 different metrics, [16] found that the selected dataset should be pre-processed to be reliably classified by the classification model. To understand how the classification is correctly performed, **confusion matrices** are effective summary tables that compare the expected and actual class labels, capturing the true positives, true negatives, false positives, and false negatives [17]. The trade-off between the true positive rate and false positive rate for a predictive model utilizing various probability thresholds is summarized by **Receiver Operating Characteristic (ROC) Curves** [18].

## 3. Methodology

This section describes the methodology for evaluating the performance of three models—BERT, RoBERTa, and RNN (LSTM)—on a sentiment classification task. Each model is fine-tuned and evaluated using a structured process. Learning rate and epochs are optimized based on suggestions in prior research [11], followed by hyperparameter tuning and performance comparison.

### 3.1 Data Pre-Processing and Model Preparation

For the most part, to ensure consistency, the same preprocessing method is used for BERT and RoBERTa, whereas a slightly different method is used for RNN (LSTM).

<p align="center">
  <img src="figures/figure2_adl.png" width="430"/>
  <br>
  <em>Figure 2: Text Length Distribution for Threads</em>
</p>

Figure 2 shows that the data is limited to threads containing up to 30 words, as the distribution predominantly falls within this range, with a few outliers exceeding 100 words. Transformer models handle these variations by applying padding to shorter sequences and truncation to longer sequences, ensuring inputs fit the specified maximum sequence length.

#### 3.1.1 RNN (LSTM)

A bidirectional LSTM with an attention mechanism is designed for sentiment classification. The data is pre-processed through various steps, including tokenization and embedding using a pre-trained Word2Vec model. Two custom classes are implemented:

- **Attention:** Implements the attention mechanism, which assigns weights to different parts of the input sequence to focus on the most relevant features for generating the output.
- **LSTMClassifier:** Integrates the attention mechanism and defines the overall model structure. It consists of an embedding layer, a bidirectional LSTM for sequence processing, and a fully connected layer for prediction. The attention mechanism processes the hidden states of the LSTM to derive a context vector, which is then used for classification.

The model is trained and validated using a stratified split of the data to maintain balanced class distributions. Oversampling is applied to address class imbalances in the training set.

#### 3.1.2 BERT and RoBERTa

For BERT and RoBERTa, text preprocessing involves several steps to clean and standardize the input data: emoji removal, URL elimination, HTML tag removal, punctuation stripping, stopword filtering, stemming, and lemmatization. After these preprocessing steps, the text is tokenized using BERT and RoBERTa's respective tokenizers, which convert the text into subword units suitable for further processing by the models.

The tokenizers encode the text and return tensors as `pt`, assigning `token_ids` and `attention_masks` relevant to both. The training samples and validation samples are created with a 2:8 ratio, providing around 36,960 training samples and 9,241 validation samples.

### 3.2 Hyperparameter Tuning

Hyperparameter tuning is performed across all three models, with strategies tailored to each architecture:

- **RNN (LSTM):** Key hyperparameters including the number of hidden units, learning rate, dropout rate, and the number of epochs are tuned.
- **BERT and RoBERTa:** Grid search is applied to explore variations in learning rates and the number of epochs.

#### 3.2.1 RNN (LSTM)

<p align="center">
  <img src="figures/trainingandvalidationlossandaccuracy.png" width="400"/>
  <br>
  <em>Figure 3: RNN Early Stopping for Best Model: Training and Validation Loss and Accuracy</em>
</p>

The LSTM model is configured with 128 hidden units, a dropout rate of 0.5, and a learning rate of 0.001 (also tested with 0.003, 0.0001). Training runs up to 10 epochs with early stopping to prevent overfitting, as seen in Figure 3. The LSTM classifier employs pre-trained Word2Vec embeddings for feature representation with a custom attention mechanism. This is particularly useful for nuanced sentiment analysis, where the context plays a critical role. For example, in the sentence "I hate that I love you," the attention mechanism can help the model focus on contrasting sentiment cues, enabling more accurate classification.

#### 3.2.2 BERT

BERT is fine-tuned with all gradients enabled, ensuring that weights in every layer are updated. Experiments are conducted with learning rates of 2×10⁻⁵ and 3×10⁻⁵, a batch size of 16, and a maximum sequence length of 128. Fine-tuning is performed over 2 or 3 epochs, involving a total of 109,491,468 parameters. In total, 4 trials are conducted.

#### 3.2.3 RoBERTa

RoBERTa follows a similar fine-tuning process, adapted to its enhanced pretraining strategy. Experiments use learning rates of 2×10⁻⁵, 3×10⁻⁵, and 5×10⁻⁵, with a batch size of 16. The maximum sequence length is set to 128, with additional tests at 64 and 32. Fine-tuning is performed over 2 or 3 epochs, involving 125,245,452 parameters. Six trials are conducted.

### 3.3 Evaluation Metrics

- **Accuracy:** Measures the overall correctness of predictions.
- **Precision, Recall, and AUROC:** Evaluated per class and averaged using macro and weighted averages, displayed in the classification report for each model.
- **Confusion Matrix:** Visualizes the distribution of correct and incorrect predictions for each class.

## 4. Results and Discussions

### 4.1 RNN Model Performance

The RNN model was evaluated with various learning rates, with the best performance at lr=0.001. Early stopping was implemented to mitigate overfitting.

<p align="center">
  <img src="figures/rnnclassreport.png" width="450"/>
  <br>
  <em>Table 1: Classification Report for RNN LSTM</em>
</p>

The LSTM model achieved an accuracy of **81.93%**, effectively classifying most cyberbullying-related tweets. However, there is room for improvement, particularly in distinguishing between closely related categories.

<p align="center">
  <img src="figures/RNNcm.png" width="400"/>
  <br>
  <em>Figure 4: Confusion Matrix for RNN LSTM</em>
</p>

The confusion matrix in Figure 4 provides a more granular view of the model's classification performance. The LSTM model correctly classifies the majority of cyberbullying categories but struggles particularly with distinguishing between the "other" and "none" categories, suggesting these two categories share similar feature distributions. Techniques such as data augmentation or more sophisticated embedding methods could help improve performance.

### 4.2 BERT Model Performance

| LR   | Epochs | Accuracy | Mean AUROC | Run Time |
|------|--------|----------|------------|----------|
| 2e-5 | 2      | 85.78%   | 97.57%     | 27:01    |
| 2e-5 | 3      | 85.38%   | 97.39%     | 40:41    |
| 3e-5 | 2      | 84.91%   | 97.48%     | 27:16    |
| 3e-5 | 3      | 85.08%   | 97.43%     | 40:57    |

*Table 2: BERT Model Performance for Different LR and Epochs*

The best classification was selected prioritizing accuracy while also choosing the model with lower Type I error than Type II error. This decision was made because there are many negative implications from being wrongfully flagged for harmful opinions, such as demonetization, social image damage, and disregarding freedom of speech if content is not actually harmful.

<p align="center">
  <img src="figures/bert_best.png" width="400"/>
  <br>
  <em>Figure 5: Confusion Matrix for BERT</em>
</p>

The confusion matrix shows balanced classification, with the biggest error observed within the "other" categories. Nearly 40% of regular threads were classified as bullying, leading to a large Type I error.

<p align="center">
  <img src="figures/bert_classification_report.png" width="450"/>
  <br>
  <em>Table 3: Classification Report for BERT</em>
</p>

The classification report shows strong performance, with "gender", "religion", "ethnicity", and "age" all classified with over 0.91 precision and over 0.85 recall. The "other" category has great recall but very low precision, signifying it incorrectly labels a higher proportion of negative instances as positives.

### 4.3 RoBERTa Model Performance

| LR   | Epochs | Accuracy | Mean AUROC | Run Time |
|------|--------|----------|------------|----------|
| 2e-5 | 2      | 84.12%   | 97.45%     | 27:10    |
| 2e-5 | 3      | 85.25%   | 97.52%     | 40:45    |
| 3e-5 | 2      | 83.20%   | 97.18%     | 27:09    |
| 3e-5 | 3      | 84.83%   | 97.45%     | 40:44    |
| 5e-5 | 2      | 83.67%   | 97.07%     | 27:08    |
| 5e-5 | 3      | 84.43%   | 97.21%     | 40:43    |

*Table 4: RoBERTa Model Performance for Different LR and Epochs*

<p align="center">
  <img src="figures/23cmrobert.png" width="400"/>
  <br>
  <em>Figure 6: Confusion Matrix for RoBERTa</em>
</p>

RoBERTa faces the same challenges as BERT but with slightly different trade-offs. There is a slightly larger Type I error for non-bullying classification. Precision for "None" is better but recall is worse—the model is cautious and avoids falsely predicting "None," but also fails to identify some true "None" cases.

<p align="center">
  <img src="figures/robertaclassificationreport.png" width="450"/>
  <br>
  <em>Table 5: Classification Report for RoBERTa</em>
</p>

The recall and F1-score appear more balanced across classes, while "None" and "others" still struggle to be identified correctly.

---

## 5. Conclusion and Future Improvements

The handling of the "others" category in data preprocessing significantly impacted model performance. Testing both BERT and RoBERTa without the "others" classification resulted in accuracy increasing to over 95% for both models. This improvement is likely due to class imbalance, which made it difficult for the models to learn from this category effectively.

Both BERT and RoBERTa achieved higher accuracy than RNN. Although RoBERTa was expected to outperform BERT, BERT achieved a slightly higher AUROC of 97.57% compared to RoBERTa's 97.52% under similar settings. Despite this, RoBERTa was more computationally expensive. In terms of classification, RoBERTa exhibited more balanced accuracy, precision, and recall across classes, whereas BERT showed a preference for minimizing Type I errors. Interestingly, a simpler model such as the RNN with LSTM outperformed both BERT and RoBERTa in recall of "None" classification.

### Future Directions

- Exploring other transformer-based models
- Implementing **Adapters** — lightweight modules added to pre-trained models to adapt to new tasks without changing the original model weights [14]
- Applying **LoRA** — low-rank decomposition to weight matrices in transformer layers for more efficient parameterization [15]
- Better preprocessing of imbalanced categories (creating subclasses or removing problematic categories)
- Moving the selected model into production for real-world cyberbullying detection

---

## References

1. Kumar, S., Roy, P. P., Dogra, D. P., & Kim, B.-G. (2023). A Comprehensive Review on Sentiment Analysis: Tasks, Approaches and Applications. *arXiv:2311.11250*. https://doi.org/10.48550/arXiv.2311.11250

2. Staudemeyer, R. C. & Morris, E. R. (2019). Understanding LSTM — a tutorial into Long Short-Term Memory Recurrent Neural Networks. *arXiv:1909.09586*. https://doi.org/10.48550/arXiv.1909.09586

3. Werbos, P. J. (1990). Backpropagation through time: what it does and how to do it. *Proceedings of the IEEE*, 78(10), 1550–1560. https://doi.org/10.1109/5.58337

4. Hochreiter, S. & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735–1780.

5. Hochreiter, S. (1991). Untersuchungen zu dynamischen neuronalen Netzen.

6. Staudemeyer, R. C. & Morris, E. R. (2019). Understanding LSTM — a tutorial into Long Short-Term Memory Recurrent Neural Networks. *arXiv:1909.09586*.

7. Wang, S. & Jiang, J. (2016). Learning Natural Language Inference with LSTM. *arXiv:1512.08849*. https://doi.org/10.48550/arXiv.1512.08849

8. Nugroho, K. S., Sukmadewa, A. Y., DW, H. W., Bachtiar, F. A., & Yudistira, N. (2021). BERT Fine-Tuning for Sentiment Analysis on Indonesian Mobile Apps Reviews. In *6th International Conference on Sustainable Information Engineering and Technology 2021* (pp. 258–264). https://doi.org/10.1145/3479645.3479679

9. Mikolov, T., Sutskever, I., Chen, K., Corrado, G., & Dean, J. (2013). Distributed Representations of Words and Phrases and their Compositionality. *arXiv:1310.4546*. https://doi.org/10.48550/arXiv.1310.4546

10. Muhammad, P., Kusumaningrum, R., & Wibowo, A. (2021). Sentiment Analysis Using Word2vec And Long Short-Term Memory (LSTM) For Indonesian Hotel Reviews. *Procedia Computer Science*. https://doi.org/10.1016/j.procs.2021.01.061

11. Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *arXiv:1810.04805*. https://doi.org/10.48550/arXiv.1810.04805

12. Nugroho, K. S. et al. (2021). BERT Fine-Tuning for Sentiment Analysis on Indonesian Mobile Apps Reviews. *(See [8])*

13. Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., Levy, O., Lewis, M., Zettlemoyer, L., & Stoyanov, V. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach. *arXiv:1907.11692*. https://doi.org/10.48550/arXiv.1907.11692

14. Houlsby, N., Giurgiu, A., Jastrzebski, S., Morrone, B., de Laroussilhe, Q., Gesmundo, A., Attariyan, M., & Gelly, S. (2019). Parameter-Efficient Transfer Learning for NLP. *arXiv:1902.00751*. https://doi.org/10.48550/arXiv.1902.00751

15. Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021). LoRA: Low-Rank Adaptation of Large Language Models. *arXiv:2106.09685*. https://doi.org/10.48550/arXiv.2106.09685

16. Vujovic, Z. (2021). Classification Model Evaluation Metrics. *International Journal of Advanced Computer Science and Applications*, 12(6). https://doi.org/10.14569/IJACSA.2021.0120670

17. Sihab-Us-Sakib, S., Rahman, M. R., Forhad, M. S. A., & Aziz, M. A. (2024). Cyberbullying detection of resource constrained language from social media using transformer-based approach. *Natural Language Processing Journal*, 9, 100104. https://doi.org/10.1016/j.nlp.2024.100104

18. Fawcett, T. (2006). Introduction to Receiver Operating Characteristics (ROC) analysis. *Pattern Recognition Letters*.
