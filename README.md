# CDSoI: Categorize the Dynamic Space of Inventions, using Empowerment 

This is a codebase that stores script implementations used to study and explore how to **Categorize the Dynamic Space of Inventions (CDSoI) using Empowerment** as a key metric. 

## Structuring of Invention Spaces 

By structuring the space of all possible Inventions/Innovations into a Multi-Dimensional space $\mathcal{I}$, where each dimension corresponds to a feature or an attribute of an Invention/Innovation, a usable Invention Space can be made. Also extendible to Novel Combination-finding in Choice-Behavior experiments.   

## Model the Invention Process as an Action-Perception Loop

1. **Actions** $A_t^n = (A_t, A_{t+1}, ..., A_{t+n-1})$: Sequences of inventive steps or modifications.

2. **Perceptions** $S_{t+n}$: Resulting state of the invention after n steps. 

## Empowerment Field  

1. Transform the Invention Space $\mathcal{I}$ into a field representing the Empowerment values of the represented Inventions. 

2. For each point i in $\mathcal{I}$, calculate empowerment $\mathfrak{E}(i)$ using:
    
    $\mathfrak{E}(i) = \max_{p(a_t^n)} \sum_{\mathcal{A}^n, \mathcal{S}} p(s_{t+n} | a_t^n) p(a_t^n) \log_2 \frac{p(s_{t+n} | a_t^n)}{\sum_{\mathcal{A}^n} p(s_{t+n} | a_t^n) p(a_t^n)}$

3. Empowerment is calculated as theorized in the paper [1], but further aided by the Semantic Approximation of the World representation using Word-Embeddings/fasttext, LLMs or other Ontological tools.

4. The calculated empowerment values are mapped across the Invention Space:
$E: \mathcal{I} \rightarrow \mathbb{R}^+$

5.  This is done over the custom built World repesentation, Gametree, Graphs, Ontologies or any other data representation methods. All built using semantic approximation of its inhabiting environment.

## Categorization of the Empowerment-Mapped Invention Space

### Empowerment Clustering

Using Clustering algorithms (e.g., k-means) to group regions of similar Empowerment values.

### Empowerment Gradient Analysis

Use Gradient Analysis to calculate $\nabla E(i)$ and then identify regions of rapid changes in the space of Empowerment values.

### Empowerment Thresholding

Define categories based on valid Empowerment ranges. e.g., (low, medium, high); (sensitivity range); (linguistic attribute categorization); (reward conditions), etc. 

## Topological Analysis

Apply **Persistent Homology** to identify topological features characterizing different regions [3].

## [Additional / Optional] Analysis

Further categorization and analysis steps. 

### Time-Dependent Empowerment

Calculate empowerment at different time steps: 

$E_{t}: \mathcal{I} \rightarrow \mathbb{R}^+$ 

Calculate empowerment at different stages of combination availabilities.

### Empowerment Flow

Define a vector field of maximum empowerment increase.

### Invention Trajectories:

Simulate paths through the invention space following empowerment gradients.

## Evaluation Metrices

1. **Predictive Power**: Assess how well empowerment-based categories predict successful inventions.

2. **Comparison with Traditional Methods**: Compare categorization results with existing invention ctegorization and classification systems.

3. **Case Studies**: Apply the methodology to well-documented historical inventions and analyze the results.

4. **Expert Validation**: Have domain experts evaluate the meaningfulness of the derived categories.

5. **Longitudinal Analysis**: Study how empowerment-based categories evolve over time in different technological domains.

6. **Refine and Iterate**: Use insights from the evaluation to refine the empowerment calculation, categorization methods, and overall methodology.

## Additional Resources

[1] And trying to update the research gaps found in the [Not by Transmission Alone](https://royalsocietypublishing.org/doi/epdf/10.1098/rstb.2020.0049) paper  by Perry et al. 

[2] Follow-up from the [Empowerment contributes to exploration behaviour in a creative video game](https://www.nature.com/articles/s41562-023-01661-2) paper by Brändle et al.

[3] About Persistent Homology [Persistent-Homology-based Machine Learning and its Applications](https://arxiv.org/abs/1811.00252).