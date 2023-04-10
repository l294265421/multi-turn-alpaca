# multi-turn-alpaca
Multi-turn alpaca is an extension of stanford alpaca and supports multi-turn dialogue. Multi-turn alpaca is trained on original alpaca data [AlpacaDataCleaned](https://github.com/gururise/AlpacaDataCleaned) and multi-turn data [ChatAlpaca](https://github.com/cascip/ChatAlpaca).

## Prapare Data
- [process_alpaca_data_cleaned](multi_turn_alpaca/prepare_data/process_alpaca_data_cleaned.py)
- [process_chat_alpaca](multi_turn_alpaca/prepare_data/process_chat_alpaca.py)
- [merge_data](multi_turn_alpaca/prepare_data/merge_data.py)

## Training Model
- [filetune](multi_turn_alpaca/training_model/finetune.py)
  - nohup sh run.sh multi_turn_alpaca/training_model/finetune.py > finetune.log 2>&1 &

## Datasets
- [stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca#fine-tuning)
- [ChatAlpaca](https://github.com/cascip/ChatAlpaca)
- [AlpacaDataCleaned](https://github.com/gururise/AlpacaDataCleaned)