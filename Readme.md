## DCSeg - Aggregation of Segmentation Models Output
Utilities to perform aggregation from multiple neural-network outputs
### Taxonomy:
    - Agent: Entity owning a segmentation model or, by extension, a segmentation model itself.
    - Proposal: The prediction of a model (segmentation), in the context of aggregation
    - Aggregation: The final output of the ensemble of agents for a particular input
    - Conflict area (also indicated as "mask" in the code): Indicates the area of the segmentation for which at least one agent gave an outcome that is different from the others.

### Aggregation methods:
- **Mean Proposal**: The aggregation is computed by averaging the predictions of the agents for each pixel
- **Weighted Mean**:
- **Pixelwise Entropy**: The weight of each pixel is the entropy of the agent outputs for the considered pixel
  - **3x3 Conv. Entropy**: The weight of each pixel is the mean entropy calculated in a 3x3 area around the pixel
  - **5x5 Conv. Entropy**: The weight of each pixel is the mean entropy calculated in a 5x5 area around the pixel
  - **Mean Entropy**: The weight of each pixel is the mean entropy of the whole image 
 - **Maximum Proposal**: The final aggregation consider only the proposal that has maximum value, for each pixel
- **Majority Voting**: The final aggregation is computed by selecting the label that is voted by most agents. If there's a tie, labels are chosen at random between the winners.
- **Negotiation**: Tool for performing multi-agent negotiation for aggregation. At each step, the agents make a proposal and each agent use a confidence function to evaluate it, then use an agent function to build a new proposal toward the final agreement. Please notice that this method is enabled by default and could slow the entire aggregation process. Consider disabiling it as it hasn't shown particular benefits for our current experiments.
    
    

### Modules:
  - Synthetic dataset generation and analysis:
     - **SyntheticSamples.py**: Base class for generating a synthetic agent
     - **run_synthetic_\<multiclass|binary>_\<pattern>_\<type>.py** Run aggregation using synthetic agents
     - **SyntheticResultViewer***: Notebooks for visualizing the results saved by the run_synthetic* script.
    
  - Segmentation on general purpose images (COCO - Animals):
     - **coco_labels.txt**: Map of integer values to label description for the COCO dataset
     - **SegmentationModel**: Base class for defining a segmentation model for the COCO dataset
     - **UnbalanceDataset_COCO**: Class for unbalancing the COCO dataset, to train segmentation models with different performances
     - **generate_coco_animals**: Script for building ground truths for the COCO dataset from their JSON representation
     - **generate_coco_predictions.py**: Script for generating the predictions for the coco dataset
     - **Generate_Base_Model**: Notebook for producing the model to fine-tune
     - **train_segmentation_models.py**: Script for fine-tuning the segmentation models on the COCO-Animals dataset
     - **evaluate_models.py**: Script for evaluating models
     
    
  - Aggregation tools:
    The aggregation module is composed by different scripts that interact with each other:
    - **Experiments.py**: Script for running all the aggregations for a specified set of predictions and ground truths. The module returns a table with various statistics (see "Result Columns") as well as the aggregations.
    - **Negotiation.py**: Script for running negotiation
    - **NegotiationTools.py**: Script that contains all the helpers for performing aggregation methods (Script name could be refactored as it doesn't reflect the content). It contains two classes for performing aggregation and various helpers to plot the results.
    - **DCSegUtils**: Various helpers (e.g. softmax implementation, feature scaling, Numpy to Pandas conversion)
    
    -run_aggregation_on_\<dataset>.py: Script for running aggregation given the predictions for the dataset
    
  - BraTS Model Aggregation:
    The BraTS segmentation models are NOT included. The predictions must be provided separately.
    Refer to the corresponding **run_aggregation*** and ***ResultViewer** files for more information.

## How to preprocess the proposals
   The aggregation method expects a LIST of numpy vectors having shape *(Agents, H, W, Labels)* for the proposals and *(H, W, Labels)* the ground truth. This means that for a binary problem you have to reshape both predictions and ground truth in such a way each channel represent a separate label. eg, a one-pixel image ground truth would have two channels: [0.0, 1.0]. A one-liner for converting  binary (single-channel) ground truth to multi-label is ``` GT = np.concatenate([1.0-GT, GT], axis=-1) ```**Feeding single-channel maps to experiments.py may lead to unexpected results**.
    
## Results columns produced by Experiments.py 
  Please Notice that some columns are only available for the binary version. Here are reported the outputs for the BraTS dataset.

   - **votes_\<Label0>_vs_\<Label1>_NvsM**: How many pixels inside the non-consensus area have been voted as \<Label0> by N agents, versus the other that voted for \<Label 1>.
   - **conflict_area**: The size (in pixel) of the conflict area. Samples with 0 conflict area should be excluded from analysis.
   - **conflict_TN**: Number of Negative pixels of the ground truth that are not part of the conflict area. 
   - **conflict_FP**: The number of pixels of conflict area that overlaps with the negative ground truth.
   - **conflict_FN**: The number of pixels of positive ground truth that are not included in the conflict area. In this case the agents already agree on a solution (either correct or wrong).
   - **conflict_TP**: The number of pixels of conflict area that overlaps with the positive ground truth. Represents the margin of improvement that is possible to obtain with an aggregation method.
   - **method**: Aggregation method used to compute the result
   - **slice**: slice number of the MRI
   - **aggregation_[full|mask]_\<metric>**: Metric calculated on the aggregation produced by "method" algorithm. If "full", all the pixels of the image have been used to compute the metric. If "mask", then only the pixel of the (positive) conflict area have been used.
   - **\<agentname>_proposal_[full|mask]_\<label>_\<metric>**: Metric calculated on the proposal (or model prediction) of the indicated agent for the indicated label. If "full", all the pixels of the image have been used to compute the metric (corresponding to the raw network output). If "mask", then only the pixel of the (positive) conflict area have been used (corresponding to the performances of the agent in the areas that are relevant for aggregation analysis).
   - **\<agentname>_votes_\<label>_[count|mean|var]**: Statistics on the Positive votes of agent \<agentname> for the label \<label>, restricted to pixels *belonging* to the conflict area. For this column, a vote is intended as True for \<label> if it is the maximum amongst all the scores in the label vector for that pixel. Count represents the number of pixels of the conflict area for which the agent voted for \<label>. Mean and variance are calculated on the corresponding probability values. They can give an insight of the average "polarization" and "stability" inside the conflict area for the indicated agent.
   - **dataset**: Indicates the SPLIT of the dataset the sample belongs to (i.e. validation or training)
   - **output_label**: Indicates the label that is considered by the coresponding ground truth
   - **patient**: patient code for the sample

   ### Evaluation metrics
   - Supported metrics: Precision, Recall, F1 score, Accuracy, Support (number of pixels for the label) 
   - Supported averaging strategies:
   	- **Micro**: Computes metrics globally by counting the total true positives, false negatives and false positives.
   	- **Macro**: Computes metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
   	- **Weighted**: Computes metrics for each label, and find their average weighted by support (the number of true instances for each label). This alters ‘macro’ to account for label imbalance; it can result in an F-score that is not between precision and recall.
