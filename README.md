# berttl

How to run the code:

## mobilebert

```
python run_glue_bias.py --data_dir=../data/glue_data/CoLA --model_type=mobilebert --model_name_or_path=google/mobilebert-uncased --task_name=cola --output_dir=../output_test --do_train --do_lower_case --memory_cost
```

## distilbert
```
python run_glue_bias.py --data_dir=../data/glue_data/CoLA --model_type=distilbert --model_name_or_path=../model/distilbert-base-uncased --task_name=cola --output_dir=../output_test --do_train --memory_cost
```


## bert
```
python run_glue_bias.py --data_dir=../data/glue_data/CoLA --model_type=bert --model_name_or_path=../model/bert_base --task_name=cola --output_dir=../output_test --do_train --memory_cost
```