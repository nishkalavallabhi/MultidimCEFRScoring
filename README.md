This repo contains the code, results, and paper for:

# Are pre-trained text representations useful for multilingual and multi-dimensional language proficiency modeling?

Reviews for the paper at BEA 2021 are also in the repo in reviews.txt.

Since we don't intend to resubmit the paper, we are leaving this version here, for anyone interested in using the code for pursuing related research. 
We may upload a version of the paper addressing reviewer concerns in future.



About the folders: 
  
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






