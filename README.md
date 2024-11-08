So this contains a lot of test notebooks and playing around. Currently active are the Graph_VAE_protein notebook.

To use there is a requirements file to install any required libs, use the prep.py to copy the smallest number of files you are looking to use.
I copied the smallest 5000 files into a directory which i used for basic testing.I also had a 100 smallest just to check dimensions of the vae were working etc.

There is then a utils file, a proteinanalyzer file which generates dfs for the relevant pdbs and a graphcreatoronehotencoder.
The graphcreatoronehotencoder creates a graph area from the dfs created by the proteinanalyzer.

There are helper  methods for ploting and printing the graphs,
I will create as well 3d plots in the protein analyzer or graphcreator...

I uploaded as well a plotly 3d in protein_notebook2, just as an example...i will create the plotting as helper functions as mentioned some place more appropriate

Feel free to add new stuff...i will do a deeper clean of the whole repo some time soon!!
