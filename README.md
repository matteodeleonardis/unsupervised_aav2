# General usage

This repository contains the code associated with the article "Unsupervised modeling of mutational landscapes of adeno-associated viruses viability". <br>
All notebooks and scripts contain Julia code. The packages necessary to run the code are contained in the project file. To install them do the following:
<ul>
    <li> move to the repository folder: <code> cd folder_path/unsupervised_aav2 </code> </li>
    <li> activate the project: <code> julia --project=. </code> </li>
    <li> instantiate the packages: <code> import Pkg; Pkg.instantiate() </code> </li>
</ul>
This code has been executed with Julia version 1.8.5. If a version >= 1.9 is used, something might not work just as it is.

# Main Files

The folder <code> my data </code> contains the data (from **[1]**) analyzed in the article in a convenient format to run the notebooks.<br>
The folder <code> preprocess and inspection </code> contains some notebooks to perform a general analysis on the data and have some important insights.
If the user doesn't need to repeat the training of the models, he can jump directly to the folder <code> analysis </code> which contains the core of the repository. <br>
The user can start from the notebook <code> fit_thresholds.ipynb </code> which has been used to estimate the threshold on the empirical log-selectivities for the binary labeling of the sequences as viable/non-viable.<br>
Then the two notebooks <code> experiment*.ipynb </code> contains the main analysis and the most important figures of the article. <br>
Finally the folder <code> logos </code> contains the notebook <code> exp3_logo.ipynb </code> that reproduces the logos plots in Fig. 7. This folder contains another Julia project file because it should be exercuted with a Julia version >= 1.9.3 to work properly.

# Secondary Files

The folders <code> phagetree_train </code> and <code> cnn_train </code> contain the script that have been used to train the two models (the former for the biophysical model and the latter for the supervised binary classificator). The trained models are respectively stored in the folders <code> phagetree_models </code> and <code> cnn_models </code> togheter with other trained models to finetune the huperparameters of the two models. The user can ignore this files because the trained model are been saved.

# Suggestions

The repository uses a couple of Julia packages that are not registered. One of them is <code> DensityPlot </code> and contains a small script to plot colored density scatter plots. It can be download and installed from **[this GitLab repo](https://gitlab.com/matteo.deleonardis2/densityplot.git)**. The other unregistered package is **[BiophysViabilityModel](https://github.com/uguzzoni/BiophysViabilityModel.git)** and contains the main implementation of the biophysical model. The packages can be manually installed via the command <code> Pkg.add("https://...") </code>. <br>

# References and external links

**[[1] Bryant, D.H., Bashir, A., Sinai, S. et al. Deep diversification of an AAV capsid protein by machine learning. Nat Biotechnol 39, 691–696 (2021). https://doi.org/10.1038/s41587-020-00793-4](https://doi.org/10.1038/s41587-020-00793-4)**
