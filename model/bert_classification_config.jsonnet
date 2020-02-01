local bert_model = "bert-base-uncased";

{
    "dataset_reader": {
        "type": "csv",
        "lazy": false,
        "tokenizer": {
            "type": "pretrained_transformer",
            "model_name": bert_model,
            "do_lowercase": true
        },
        "token_indexers": {
            "bert": {
                "type": "bert-pretrained",
                "pretrained_model": bert_model,
            }
        }
    },
    "train_data_path": "data/2014_Middle_East_Respiratory_Syndrome_en/2014_mers_cf_labels_train.csv",
    "validation_data_path": "data/2014_Middle_East_Respiratory_Syndrome_en/2014_mers_cf_labels_test.csv",
    "model": {
        "type": "bert_for_classification",
        "bert_model": bert_model,
        "dropout": 0.1,
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys": [["tokens", "num_tokens"]],
        "batch_size": 32
    },
    "trainer": {
        "optimizer": {
            "type": "bert_adam",
            "lr": 2e-5
        },
        "validation_metric": "+accuracy",
        "num_serialized_models_to_keep": 1,
        "num_epochs": 10,
        "grad_norm": 1.0,
        "cuda_device": 0
    }
}