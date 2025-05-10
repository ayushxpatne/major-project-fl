so we have a run.sh which is basically a script file to start venv, make directories and make clients 

now we have venv path, and since python executable command was not working we need to explictily  call the the python 
executable which is present inside the venv folder

now it first checks if client 1 train csv exists if exists if not then we need to call data_split/py which splits 
the dataset into 3 datasets with varying ratios of fraud in them, 3 cause we are simulatig 3 clients. 

now how the cleaning works. 

we found this dataset on kaggle : https://www.kaggle.com/datasets/vagifa/ethereum-frauddetection-dataset/data 
which consisted of around 9k transactions of etheurum, marks as fraud or no, 
fraud counts before preprocessing :
flag:
0    7662
1    2179

flag is the column which defines if transaction is fraud or not. 
so far as you can see the data is imbalanced. 

**Lets talk about Data Exploration and Cleaning:**

we are using numpy, pandas, seaborn and matplotlib.

the dataset description is as follows:
# Dataset Description

| **Feature**                                              | **Description**                                                                |
| -------------------------------------------------------- | ------------------------------------------------------------------------------ |
| **Index**                                                | The index number of a row.                                                     |
| **Address**                                              | The address of the Ethereum account.                                           |
| **FLAG**                                                 | Whether the transaction is fraud or not.                                       |
| **Avg min between sent tnx**                             | Average time between sent transactions for the account in minutes.             |
| **Avg_min_between_received_tnx**                         | Average time between received transactions for the account in minutes.         |
| **Time_Diff_between_first_and_last(Mins)**               | Time difference between the first and last transaction.                        |
| **Sent_tnx**                                             | Total number of sent normal transactions.                                      |
| **Received_tnx**                                         | Total number of received normal transactions.                                  |
| **Number_of_Created_Contracts**                          | Total number of created contract transactions.                                 |
| **Unique_Received_From_Addresses**                       | Total unique addresses from which the account received transactions.           |
| **Unique_Sent_To_Addresses**                             | Total unique addresses to which the account sent transactions.                 |
| **Min_Value_Received**                                   | Minimum value in Ether ever received.                                          |
| **Max_Value_Received**                                   | Maximum value in Ether ever received.                                          |
| **Avg_Value_Received**                                   | Average value in Ether ever received.                                          |
| **Min_Val_Sent**                                         | Minimum value of Ether ever sent.                                              |
| **Max_Val_Sent**                                         | Maximum value of Ether ever sent.                                              |
| **Avg_Val_Sent**                                         | Average value of Ether ever sent.                                              |
| **Min_Value_Sent_To_Contract**                           | Minimum value of Ether sent to a contract.                                     |
| **Max_Value_Sent_To_Contract**                           | Maximum value of Ether sent to a contract.                                     |
| **Avg_Value_Sent_To_Contract**                           | Average value of Ether sent to contracts.                                      |
| **Total_Transactions(Including_Tnx_to_Create_Contract)** | Total number of transactions.                                                  |
| **Total_Ether_Sent**                                     | Total Ether sent for the account address.                                      |
| **Total_Ether_Received**                                 | Total Ether received for the account address.                                  |
| **Total_Ether_Sent_Contracts**                           | Total Ether sent to contract addresses.                                        |
| **Total_Ether_Balance**                                  | Total Ether balance following enacted transactions.                            |
| **Total_ERC20_Tnxs**                                     | Total number of ERC20 token transfer transactions.                             |
| **ERC20_Total_Ether_Received**                           | Total ERC20 token received transactions in Ether.                              |
| **ERC20_Total_Ether_Sent**                               | Total ERC20 token sent transactions in Ether.                                  |
| **ERC20_Total_Ether_Sent_Contract**                      | Total ERC20 token transfer to other contracts in Ether.                        |
| **ERC20_Uniq_Sent_Addr**                                 | Number of ERC20 token transactions sent to unique account addresses.           |
| **ERC20_Uniq_Rec_Addr**                                  | Number of ERC20 token transactions received from unique addresses.             |
| **ERC20_Uniq_Rec_Contract_Addr**                         | Number of ERC20 token transactions received from unique contract addresses.    |
| **ERC20_Avg_Time_Between_Sent_Tnx**                      | Average time between ERC20 token sent transactions in minutes.                 |
| **ERC20_Avg_Time_Between_Rec_Tnx**                       | Average time between ERC20 token received transactions in minutes.             |
| **ERC20_Avg_Time_Between_Contract_Tnx**                  | Average time between ERC20 token sent transactions to contracts.               |
| **ERC20_Min_Val_Rec**                                    | Minimum value in Ether received from ERC20 token transactions for the account. |
| **ERC20_Max_Val_Rec**                                    | Maximum value in Ether received from ERC20 token transactions for the account. |
| **ERC20_Avg_Val_Rec**                                    | Average value in Ether received from ERC20 token transactions for the account. |
| **ERC20_Min_Val_Sent**                                   | Minimum value in Ether sent from ERC20 token transactions for the account.     |
| **ERC20_Max_Val_Sent**                                   | Maximum value in Ether sent from ERC20 token transactions for the account.     |
| **ERC20_Avg_Val_Sent**                                   | Average value in Ether sent from ERC20 token transactions for the account.     |
| **ERC20_Uniq_Sent_Token_Name**                           | Number of unique ERC20 tokens transferred.                                     |
| **ERC20_Uniq_Rec_Token_Name**                            | Number of unique ERC20 tokens received.                                        |
| **ERC20_Most_Sent_Token_Type**                           | Most sent token for the account via ERC20 transactions.                        |
| **ERC20_Most_Rec_Token_Type**                            | Most received token for the account via ERC20 transactions.                    |
---

we dropped few columns which do not provide any useful information.
df = df.drop(columns=['Unnamed: 0', 'Address', 'Index'], errors='ignore')  
then we checked for some missing values and data types and found percentage of missing values in each column.

and we found out
Total ERC20 tnxs                        8.423941
ERC20 total Ether received              8.423941
ERC20 total ether sent                  8.423941
ERC20 total Ether sent contract         8.423941
ERC20 uniq sent addr                    8.423941
ERC20 uniq rec addr                     8.423941
ERC20 uniq sent addr.1                  8.423941
ERC20 uniq rec contract addr            8.423941
ERC20 avg time between sent tnx         8.423941
ERC20 avg time between rec tnx          8.423941
ERC20 avg time between rec 2 tnx        8.423941
ERC20 avg time between contract tnx     8.423941
ERC20 min val rec                       8.423941
ERC20 max val rec                       8.423941
ERC20 avg val rec                       8.423941
ERC20 min val sent                      8.423941
ERC20 max val sent                      8.423941
ERC20 avg val sent                      8.423941
ERC20 min val sent contract             8.423941
ERC20 max val sent contract             8.423941
ERC20 avg val sent contract             8.423941
ERC20 uniq sent token name              8.423941
ERC20 uniq rec token name               8.423941
ERC20 most sent token type             27.405751
ERC20_most_rec_token_type               8.850727

and we Check the amount of classes in our categorical features:
ERC20 most sent token type
0                                                         4399
                                                          1191
EOS                                                        138
OmiseGO                                                    137
Golem                                                      130
                                                          ... 
BlockchainPoland                                             1
Covalent Token                                               1
Nebula AI Token                                              1
Blocktix                                                     1
eosDAC Community Owned EOS Block Producer ERC20 Tokens       1
Name: count, Length: 304, dtype: int64

and

ERC20_most_rec_token_type
0                        4399
OmiseGO                   873
Blockwell say NOTSAFU     779
DATAcoin                  358
Livepeer Token            207
                         ... 
BCDN                        1
Egretia                     1
UG Coin                     1
Yun Planet                  1
INS Promo1                  1
Name: count, Length: 466, dtype: int64

and then we plotted out distribution graphs of each column/feature, 
here we found out features which have no variance
the featurses were:['ERC20 avg time between sent tnx',
 'ERC20 avg time between rec tnx',
 'ERC20 avg time between rec 2 tnx',
 'ERC20 avg time between contract tnx',
 'ERC20 min val sent contract',
 'ERC20 max val sent contract',
 'ERC20 avg val sent contract']

 now we cleaned data by 
 A. replacing empty columns by zeros,
 B. dropping categorical columns as they may drastically increase the complexity of our model without providing significantly useful information.
 C. Drop columns with no variance
 D. Columns with small distribution (if len(df[i].value_counts()) < 10 ), and the columns were 'min value sent to contract','total ether sent contracts','ERC20 uniq sent addr.1'

 so finally the dataset was cleaned and we are left with 36 columss, compared to 50 columns before cleaning.
 
 now we selected numerical colums ie the remeaining columsn and plotted a heatmap / correlation matrix to see the correlation between the features.
[correlation_matrix_numerical_features.png]

to detect highly correalted pairs, we kept the threshold as 0.85
the output was :
avg value sent to contract + max val sent to contract: 0.95
ERC20 max val rec + ERC20 total Ether received: 1.00
ERC20 avg val rec + ERC20 total Ether received: 0.86
ERC20 avg val rec + ERC20 max val rec: 0.86
ERC20 min val sent + ERC20 total ether sent: 1.00
ERC20 max val sent + ERC20 total ether sent: 1.00
ERC20 max val sent + ERC20 min val sent: 1.00
ERC20 avg val sent + ERC20 total ether sent: 1.00
ERC20 avg val sent + ERC20 min val sent: 1.00
ERC20 avg val sent + ERC20 max val sent: 1.00
ERC20 uniq rec token name + ERC20 uniq rec contract addr: 1.00

now we dropped the highly correlated features ie  'avg value sent to contract',  
    'max val sent to contract',
    'ERC20 max val rec',
    'ERC20 avg val rec',
    'ERC20 min val sent',
    'ERC20 max val sent',
    'ERC20 avg val sent',
    'ERC20 uniq rec token name',

we again plot the correaltion matrix [correlation_matrix_numerical_features_after_cleaning.png]

after ensuring there are no null values we are now accounting for the imbalanced dataset.
we are using the Smoteen algorithm to balance the datase ie Over-sample the minority class using SMOTE and clean with ENN (Edited Nearest Neighbors)

after balancing the dataset has 12353 rows,  27 features/colums in train, and 1 column in test. ie X and y.

we save this datasets as global_train_centralized.csv and global_test_centralized.csv.

now once we have the cleaned dataset we move on to the federated system creation.

Next Step is to first split the training dataset into 3 parts for 3 clients we have.

so we are splitting the dataset into 3 parts for 3 clients in ratio:
CLIENT_FRAUD_CONFIG = {
    0: {'fraud_ratio': 0.04, 'name': 'client_1'},
    1: {'fraud_ratio': 0.06, 'name': 'client_2'},
    2: {'fraud_ratio': 0.08, 'name': 'client_3'}
}
The varying fraud ratios across clients in  federated learning (FL) setup simulate real-world scenarios where different entities (exchanges, wallets, etc.) have different risk profiles. 

1. Why Use Different Fraud Ratios?
A. Real-World Non-IIDness
In Ethereum/crypto:

Some exchanges (e.g., regulated ones) have low fraud rates (e.g., 1-2%).

Others (e.g., shady platforms) might have higher fraud rates (e.g., 5-10%).

Your split mimics this:

client_1 (4% fraud): Represents a low-risk exchange.

client_3 (8% fraud): Represents a high-risk marketplace.

B. Test FL’s Robustness
FL must handle imbalanced data across clients.

If all clients had identical fraud ratios (IID), FL would behave like centralized ML (boring!).

Non-IID splits reveal if the global model:

Biases toward high-fraud clients (e.g., overfits to client_3).

Generalizes well despite uneven data.

C. Edge Cases Matter
A model trained only on 1% fraud data might miss subtle fraud patterns.

Including a high-fraud client helps the global model learn stronger fraud signals.

we also implementedd a non-iid split for the test dataset. iid stands for independently and identically distributed, which means that the data is distributed in a way that is independent of the order in which it is collected, ie no row is repeated
in the split.
now we are ready to create the federated learning system.

Lets start with server.py

we are mainly using flower to create and manage federated learning system ie client nodes and center. 

lets start with few parameters/global variables for this file:
NUM_ROUNDS = 10 => how many rounds of training we want to do for eeach client
AGGREGATE_EVERY_N_ROUNDS = 3  # This is for metrics evaluation and model saving, not aggregation of weights
MIN_AVAILABLE_CLIENTS = 2
MIN_FIT_CLIENTS = 2
MIN_EVALUATE_CLIENTS = 2


1. we get the number of model input features from the dataset.
- so basically when we oversampled and then normalised the dataset we saved it as joblib file so that we can access the 
  it in the server to create a pipeline for the FL system. 
- the get_model_input_features() from model.py is used to get the number of input features from the dataset.

2. now we define strategy for the federated learning system in flower: 
   # Define the Flower strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.8,  # Sample 80% of available clients for training
        fraction_evaluate=0.8,  # Sample 80% of available clients for evaluation
        min_fit_clients=MIN_FIT_CLIENTS,
        min_evaluate_clients=MIN_EVALUATE_CLIENTS,
        min_available_clients=MIN_AVAILABLE_CLIENTS,
        evaluate_fn=get_evaluate_fn(FraudDetectionNet, num_features),
        on_fit_config_fn=lambda rnd: {"epoch": 5, "batch_size": 32},  # 5 epochs per round
    )
    # Start the Flower server
    fl.server.start_server(
        server_address="0.0.0.0:3000",
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )

once server is started we will start the client nodes.
we are using the client.py file to create the client nodes.

to generalise, 
a client when created calls the FraudeDetectionNet class from model.py, loads the respective train on local dataset.




