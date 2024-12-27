# VirtualGeneSuppression_Preprocessing
This repository presents a novel preprocessing tool for Non-Negative Matrix Factorization of scRNAseq. Due to their non-negativity, the NMF metafactors have interpretation as 'metagenes', and have been shown to correspond well to known biological pathways which are sources of heterogeneity among the cells in our data, e.g. Apoptosis in cancer cells. Meanwhile, the genes which are largely co-expressed in these pathways have also been shown to form clusters among the cells. Therefore, by clustering the genes and iteratively suppressing each cluster before fitting the NMF model, we can force each model to seek out alternative sources of biological variation, leading to a more informative and robust representation. This, in turn, improves performance for cell type identification via clustering, as well as dimensionality reduction for visualization.

You guys may try to augment this code by considering alternative approaches to grouping the genes, such as examining their co-expression among all the cells. Currently the code concatenates the outputs of the different NMF models. You may also try a consensus clustering approach. Lastly, you may try to use various regularized NMF variants or other dimensionality reduction techniques to showcase the generalizability of the technique. Let me know if you have any questions. 
