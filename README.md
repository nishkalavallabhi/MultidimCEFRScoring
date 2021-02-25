This repo contains the code, results, and paper for:

# Are pre-trained text representations useful for multilingual and multi-dimensional language proficiency modeling?

**Abstract:**  
Development of language proficiency models for non-native learners has been an active area of interest in NLP research for the past few years. Although language proficiency is multidimensional in nature, existing research typically considers a single *overall proficiency* while building models. Further, existing approaches also considers only one language at a time. This paper describes our experiments and observations about the role of pre-trained and fine-tuned multilingual embeddings in performing multi-dimensional, multilingual language proficiency classification. We report experiments with three languages -- German, Italian, and Czech -- and model seven dimensions of proficiency ranging from vocabulary control to sociolinguistic appropriateness. Our results indicate that while fine-tuned embeddings are useful for multilingual proficiency modeling, none of the features achieve consistently best performance for all dimensions of language proficiency.

Reviews for the paper from [BEA 2021](https://sig-edu.org/bea/current) are also in the repo. Since we don't intend to resubmit the paper, we are leaving this version here, for anyone interested in using the code (the version submitted as supplementary material for the submission) for pursuing related research.  We may upload a version of the paper addressing reviewer concerns in future.

**About the folders: **
  
- **Dataset/:**  
  * This contains folders with learner corpora for three languages(German-DE, Italian-IT, Czech-CZ), from [MERLIN project](http://www.merlin-platform.eu/), and their dependency parsed versions (DE-Parsed, IT-Parsed, CZ-Parsed) using [UDPipe](http://ufal.mff.cuni.cz/udpipe).  
  * The [original MERLIN corpus](http://www.merlin-platform.eu/C_data.php) is available under a [Creative Commons Attribution-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-sa/4.0/), which allows us to: "copy and redistribute the material in any medium or format".  
  * The original corpus files have been renamed for our purposes, and all metadata showing information about the learner background, different dimensions of proficiency etc have been removed. All .Metadata.txt files in this folder contain information about other CEFR proficiency dimensions for the learner writings, which we did not use in this paper.  
  * RemovedFiles/ folder in this folder contains files that we did not use for our analysis, either because they are unrated, or because that proficiency level has too few examples (Read the paper for details).  
  
- **code/:**
  * Baseline.py: contains for ngram features, all dimensions  
  * doclenbaseline.py: contains document length baseline, all dimensions   
  * gen_laser_vec.py, gen_mbert_vecs.py: contain code for generating LASER and mBERT features (not fine tuned)  
  * LaserBertExperiments.py: Fine tuning code.  

- **ResultFile/:**
  * This folder contains the detailed result files generated while doing the experiments, primarily to keep a record. It is useful for replication and comparison purposes in future.  

- **README.md**: This file  

- **requirements.txt:** Generated using pipreqs






