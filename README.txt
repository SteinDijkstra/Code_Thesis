Project: Predicting Pedestrian Movement Paths using Evidential Neural Networks 
Author: Stein Dijkstra

Structure codebase:
- Code:
    main.py: File which executes the code programmed in the other folders
    /data_preparation: Contains the code to generate the regression data and to preprocess the pedestrian dataset
    /models: Contains the implementation of the evidential, adapted evidential, MC dropout and ensemble models
    /experiment: Contains the code that organizes the experiments used in the thesis, also contains a file that helps analyze the results
    /utility: Utility methods related to the config file

- Archive:
    /data:
        /ETH_data: The ETH Data retrieved from Amirian, J., Zhang, B., Castro, F. V., Baldelomar, J. J., Hayet, J.-B., & Pettre, J. (2020). Opentraj
        /UCY_data: The UCY Data retrieved from Amirian, J., Zhang, B., Castro, F. V., Baldelomar, J. J., Hayet, J.-B., & Pettre, J. (2020). Opentraj
        /Preprocessed: folder with processed regression, ETH and UCY data according to the data section in the thesis
    /results: folder with different runs with stored results
    /figures: folder with the figures presented in the thesis

- Other:
    notebooks: contains jupyter notebook to create the tables and figures in the thesis
    config: contains the config file that changes the architecture of the network, structure needs to be specified in  utility/config.py

How to run:
    1. Create a conda environment according to requirements/environment.yaml
    2. Change path in /utility/base/
    2. Change/run main.py or the notebook

    