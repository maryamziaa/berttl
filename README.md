# berttl

How to run the code:

## Requirements
Python 3.6.3
pip install -r requirements.txt

## Download GLUE data

Go to the data directory by ```cd data```

Download GLUE dataset, CoLA task using ```python dl_glue_data.py --data_dir glue_data --tasks CoLA```.

## Download distilbert model

Go to the model directory by ```cd model```

Download Distilbert_base model by ```python download_distilbert-base-uncased.py```.

## mobilebert

Updating biases and freezing weights in FFN layers:
```
python run_glue_bias.py --data_dir=../data/glue_data/CoLA --model_type=mobilebert --model_name_or_path=google/mobilebert-uncased --task_name=cola --output_dir=../output_test --do_train --do_eval --do_lower_case --memory_cost
```

Adding the lite residual network te replace FFN layers:

```
python run_glue_lite.py --data_dir=../data/glue_data/CoLA --model_type=mobilebert --model_name_or_path=google/mobilebert-uncased --task_name=cola --output_dir=../output_test --do_train --do_eval --do_lower_case --memory_cost
```

## distilbert

updating biases and freezing weights in FFN layers:
```
python run_glue_bias.py --data_dir=../data/glue_data/CoLA --model_type=distilbert --model_name_or_path=../model/distilbert-base-uncased --task_name=cola --output_dir=../output_test --do_train --do_eval --memory_cost
```

Adding the lite residual network te replace FFN layers:
```
python run_glue_lite.py --data_dir=../data/glue_data/CoLA --model_type=distilbert --model_name_or_path=../model/distilbert-base-uncased --task_name=cola --output_dir=../output_test --do_train --do_eval --memory_cost
```


