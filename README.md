# Time Robust Trees: Using Temporal Invariance to Improve Generalization

Experiments' code and data for the `Time Robust Trees: Using Temporal Invariance to Improve Generalization` paper.

For the algorithm code, see [Time Robust Forest package](https://github.com/lgmoneda/time-robust-forest).

## Code

1. Use the file `environment.yml` to create a new conda environment (`conda env create -f environment. yml`)
2. Download the datasets
3. Execute the notebooks

The paper image will be saved in the `images/`. Notice the data volume is large and the  time-robust-forest version used (v0.1.13) is slow, so using 32-64 cores is recommended.

## Datasets

The 20 news group dataset can be loaded directly from sklearn, check the notebook. All the other datasets can be retrieved from Kaggle datasets:

Chicago, C.: Chicagocrime-bigquerydataset(2021), version 1. Retrieved March
13, 2021 from https://www.kaggle.com/chicago/chicago-crime
Daoud, J.: Animal shelter dataset (2021), version 1. Retrieved March 13, 2021 from
https://www.kaggle.com/jackdaoud/animal-shelter-analytics
Moneda, L.: Globo esporte news dataset (2020), version 18. Retrieved March 31, 2021 from https://www.kaggle.com/lgmoneda/ge-soccer-clubs-news
Mouill, M.: Kickstarter projects dataset (2018), version 7. Retrieved March
13, 2021 from https://www.kaggle.com/kemical/kickstarter-projects?select=ks-projects-201612.csv
Shastry, A.: San francisco building permits dataset (2018), version 7. Retrieved March 13, 2021 from https://www.kaggle.com/aparnashastry/building-permit-applications-data
Sionek, A.: Brazilian e-commerce public dataset by olist (2019), version 7. Re- trieved March 13, 2021 from https://www.kaggle.com/olistbr/brazilian-ecommerce
