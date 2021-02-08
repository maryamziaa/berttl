"""
PyTorch Profiler
====================================
This recipe explains how to use PyTorch profiler and measure the time and
memory consumption of the model's operators.

Introduction
------------
PyTorch includes a simple profiler API that is useful when user needs
to determine the most expensive operators in the model.

In this recipe, we will use a simple Resnet model to demonstrate how to
use profiler to analyze model performance.

Setup
-----
To install ``torch`` and ``torchvision`` use the following command:

::

   pip install torch torchvision


"""


######################################################################
# Steps
# -----
#
# 1. Import all necessary libraries
# 2. Instantiate a simple Resnet model
# 3. Use profiler to analyze execution time
# 4. Use profiler to analyze memory consumption
# 5. Using tracing functionality
#
# 1. Import all necessary libraries
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# In this recipe we will use ``torch``, ``torchvision.models``
# and ``profiler`` modules:
#

import argparse
from tqdm import tqdm

import torch
import torchvision.models as models
import torch.autograd.profiler as profiler
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)


from transformers import (BertConfig,
                                  BertForSequenceClassification, BertTokenizer,
                                  DistilBertConfig,
                                  DistilBertForSequenceClassification,
                                  DistilBertTokenizer,
                                  MobileBertConfig, MobileBertForSequenceClassification, MobileBertTokenizer
                                )
from transformers import glue_processors as processors
from transformers import glue_output_modes as output_modes
from transformers import glue_convert_examples_to_features as convert_examples_to_features


parser = argparse.ArgumentParser()
parser.add_argument("--model_type", default=None, type=str, required=True)
parser.add_argument("--model_name_or_path", default=None, type=str, required=True)
parser.add_argument("--task_name", default=None, type=str, required=True)
parser.add_argument("--config_name", default="", type=str)
parser.add_argument("--cache_dir", default="", type=str)
parser.add_argument("--tokenizer_name", default="", type=str)
parser.add_argument("--do_lower_case", action='store_true')
parser.add_argument("--max_seq_length", default=128, type=int)
args = parser.parse_args()

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
    'mobilebert': (MobileBertConfig, MobileBertForSequenceClassification, MobileBertTokenizer)
}


# Prepare GLUE task
args.task_name = args.task_name.lower()
if args.task_name not in processors:
    raise ValueError("Task not found: %s" % (args.task_name))
processor = processors[args.task_name]()
args.output_mode = output_modes[args.task_name]
label_list = processor.get_labels()
num_labels = len(label_list)


config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          num_labels=num_labels,
                                          finetuning_task=args.task_name,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
model = model_class.from_pretrained(args.model_name_or_path,
                                        from_tf=bool('.ckpt' in args.model_name_or_path),
                                        config=config,
                                        cache_dir=args.cache_dir if args.cache_dir else None)

#print(model)

######################################################################
# 2. Instantiate a bert model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Let's create an instance of a bert model and prepare an input
# for it:
#

inputs = {'input_ids':      torch.randint(0, 2843, (1,128)).long(),
            'attention_mask': torch.randint(0, 2, (1,128)).long(),
            'labels':         torch.randint(0, 1, (1,)).long()}


'''  
inputs['input_ids'].shape= torch.Size([b_size, max_seq_length]
inputs['attention_mask'].shape=torch.Size([b_size, max_seq_length])
inputs['labels'].shape = torch.Size([b_size]
'''

######################################################################
# 3. Use profiler to analyze execution time
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# PyTorch profiler is enabled through the context manager and accepts
# a number of parameters, some of the most useful are:
#
# - ``record_shapes`` - whether to record shapes of the operator inputs;
# - ``profile_memory`` - whether to report amount of memory consumed by
#   model's Tensors;
# - ``use_cuda`` - whether to measure execution time of CUDA kernels.
#
# Let's see how we can use profiler to analyze the execution time:

with profiler.profile(record_shapes=True) as prof:
    with profiler.record_function("model_inference"):
        #model(inputs)
        model(**inputs)


######################################################################
# Note that we can use ``record_function`` context manager to label
# arbitrary code ranges with user provided names
# (``model_inference`` is used as a label in the example above).
# Profiler allows one to check which operators were called during the
# execution of a code range wrapped with a profiler context manager.
# If multiple profiler ranges are active at the same time (e.g. in
# parallel PyTorch threads), each profiling context manager tracks only
# the operators of its corresponding range.
# Profiler also automatically profiles the async tasks launched
# with ``torch.jit._fork`` and (in case of a backward pass)
# the backward pass operators launched with ``backward()`` call.
#
# Let's print out the stats for the execution above:

print(prof.key_averages().table(sort_by="cpu_time_total"))


######################################################################
# Note the difference between self cpu time and cpu time - operators can call other operators, self cpu time exludes time
# spent in children operator calls, while total cpu time includes it.
#
# To get a finer granularity of results and include operator input shapes, pass ``group_by_input_shape=True``:

print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total"))

######################################################################
# 4. Use profiler to analyze memory consumption
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# PyTorch profiler can also show the amount of memory (used by the model's tensors)
# that was allocated (or released) during the execution of the model's operators.
# In the output below, 'self' memory corresponds to the memory allocated (released)
# by the operator, excluding the children calls to the other operators.
# To enable memory profiling functionality pass ``profile_memory=True``.

with profiler.profile(profile_memory=True, record_shapes=True) as prof:
    model(**inputs)

print(prof.key_averages().table(sort_by="self_cpu_memory_usage"))


print(prof.key_averages().table(sort_by="cpu_memory_usage"))

######################################################################
# 5. Using tracing functionality
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Profiling results can be outputted as a .json trace file:

with profiler.profile() as prof:
    with profiler.record_function("model_inference"):
        model(**inputs)

prof.export_chrome_trace("profilers/"+args.model_type+"_inference_trace.json")

######################################################################
# User can examine the sequence of profiled operators after loading the trace file
# in Chrome (``chrome://tracing``):
#
# .. image:: ../../_static/img/trace_img.png
#    :scale: 25 %

######################################################################
