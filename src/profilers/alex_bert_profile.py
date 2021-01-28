import torch 
import torch.nn as nn

from transformers import (
    BertConfig,
    # BertForSequenceClassification,
    BertTokenizer,
)
# from modeling_profiled_bert import BertForSequenceClassification, BertLayer
from transformers import BertForSequenceClassification, BertLayer
from transformers.models.distilbert import DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer
from transformers.models.distilbert.modeling_distilbert import TransformerBlock
MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer, BertLayer, 'intermediate_size', 'num_hidden_layers'),
    'distilbert': (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer, TransformerBlock, 'hidden_dim', 'n_layers'),
}
# MODEL_DIM_NAMES = {
#     'bert': ,
#     'distilbert': 'hidden_dim'
# }

# from bert_flex import (
#     BertForSequenceClassification as BertForSequenceClassificationFlex,
#     BertConfig as BertConfigFlex,
# )
# import torchprof
# import torch.autograd.profiler as profiler
import cProfile
# import pstats
# from pstats import SortKey

import argparse
import os

from transformers import glue_compute_metrics as compute_metrics
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from parser import build_args

import time, datetime
import numpy as np

def load_and_cache_examples(args, task, tokenizer, evaluate=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}".format(
            "dev" if evaluate else "train",
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
            str(task),
        ),
    )
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        print("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        # logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if task in ["mnli", "mnli-mm"] and args.model_type in ["roberta", "xlmroberta"]:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        examples = (
            processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(args.data_dir)
        )
        features = convert_examples_to_features(
            examples,
            tokenizer,
            max_length=args.max_seq_length,
            label_list=label_list,
            output_mode=output_mode,
        )
        if args.local_rank in [-1, 0]:
            print("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset


def process_stats(filename):
    p = pstats.Stats(filename)
    p = p.strip_dirs()
    p.sort_stats(SortKey.CUMULATIVE) # sort by cum time, print top 10
    p.print_stats('modeling_profiled_bert', 10)

    # p.print_stats('forward',10) # only forward methods, top 10
    # p.print_callees('forward',10)

def evaluate(args, model, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + "/MM") if args.task_name == "mnli" else (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
        
        for i,batch in enumerate(tqdm(eval_dataloader, desc="Evaluating")):
            model.eval()
            if i >= args.n_trials:
                break
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                if args.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2] if args.model_type in ["bert", "masked_bert", "xlnet", "albert"] else None
                    )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
                # with profiler.profile(record_shapes=True) as prof:
                #     with profiler.record_function("model_inference"):
                # print('before  ',torch.cuda.memory_stats(args.device)['allocated_bytes.all.peak']/(2**20))
                outputs = model(**inputs)
    return 0
    
def evaluate_autograd_profiler(args, model, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + "/MM") if args.task_name == "mnli" else (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
        
        paths = [("BertForSequenceClassification", "bert", "encoder","layer","1"), ("BertForSequenceClassification", "bert", "encoder","layer","1","attention"),("BertForSequenceClassification", "bert", "encoder","layer","1","intermediate","dense"), ("BertForSequenceClassification", "bert", "encoder","layer","1","output","dense")]
        
        for i,batch in enumerate(tqdm(eval_dataloader, desc="Evaluating")):
            model.eval()
            if i >= args.n_trials:
                break
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                if args.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2] if args.model_type in ["bert", "masked_bert", "xlnet", "albert"] else None
                    )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
                # with profiler.profile(record_shapes=True) as prof:
                #     with profiler.record_function("model_inference"):
                # torch.cuda.synchronize()
                with torchprof.Profile(model, use_cuda=False, paths=paths) as prof:
                    outputs = model(**inputs)
        # print(prof.display(show_events=False))
        prof_str, prof_stats = prof.display(show_events=False)
        return prof_str, prof_stats
# getting "RuntimeError: Profiler is already enabled on this thread"  error and idk what to do. It happened after I added a torch.num_threads line. But removed line and it still sucks

def cprof_main():
    args=build_args()    

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # setup for latency test:
    # args.device='cpu' # jsut want to test cpu for now
    args.model_name_or_path = 'bert-base-uncased'
    args.per_gpu_eval_batch_size=1
    argdict = vars(args)
    # torch.set_num_threads(32)

    
    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

   
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = BertConfig, BertForSequenceClassification, BertTokenizer
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
        do_lower_case=args.do_lower_case,
    )
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        # cache_dir=args.cache_dir if args.cache_dir else None,
        num_hidden_layers=12,
        # pruning_method=args.pruning_method,
        # mask_init=args.mask_init,
        # mask_scale=args.mask_scale,
    )
   

    model = model_class(
        # args.model_name_or_path,
        # from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        # cache_dir=args.cache_dir if args.cache_dir else None,
    )
    # print('hello0')
    model.to(args.device)
    evaluate(args, model, tokenizer)
    # torch.autograd.profiler.profile(enabled=False)
    # prof_str, prof_stats = evaluate_autograd_profiler(args,model,tokenizer)

    # make a dataframe with each column named according to cpu statistic
        # each row corresponds to dff dimension
        # fill rows for each dimension in for loop
    print(prof_stats)

    tested_dims=[]
    columns = []
    latencies = {} # want to reuse keys of prof_stats
    for key in prof_stats.keys():
        latencies[key]=[]
        columns.append(str(key))
    latencies['dff'] = []
   
    min_dim=768
    max_dim=768*4
    for dff in range(min_dim,min_dim+17,16):
        tested_dims.append(dff)

        config.intermediate_size = dff
        model = model_class(config=config)
        model.to(args.device)
        prof_str, prof_stats = evaluate_autograd_profiler(args,model,tokenizer)

        
        for key in prof_stats.keys(): # for each module examined, store the cpu time
            latencies[key].append(prof_stats[key].cpu_total/args.n_trials)
        latencies['dff'].append(dff)
    print(latencies)
    
    # latency_dataframe = pd.DataFrame.from_dict(data=latencies, orient='columns')
    print(latency_dataframe)
    print(latency_dataframe[('BertForSequenceClassification', 'bert', 'encoder', 'layer', '1', 'intermediate', 'dense')])


import gc
def latency_measurement():
    args=build_args()    

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device
    # args.model_name_or_path = 'bert-base-uncased'    
    
    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class, model_layer, ffn_dim_name, n_layers_name = MODEL_CLASSES[args.model_type]
    # ffn_dim_name = MODEL_DIM_NAMES[args.model_type]
    # config_class, model_class, tokenizer_class = BertConfig, BertForSequenceClassification, BertTokenizer
    # config_class, model_class, tokenizer_class = BertConfigFlex, BertForSequenceClassificationFlex, BertTokenizer

    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        # num_hidden_layers=args.num_hidden_layers,
        torchscript=True, # this is keyyyyyyyy. distilbert config does not like me setting this
        # intermediate_size_layers=[3072]*args.num_hidden_layers
    )
    if args.num_hidden_layers is not None:
        setattr(config, n_layers_name, args.num_hidden_layers)

    if args.num_threads: # if not 0
        torch.set_num_threads(args.num_threads) # single threaded latencies are slower but have much lower std dev so better for testing (since xeon CPU is not realistic scenario anyways)

    # process results
    dt = datetime.datetime.now()
    dt_string = dt.strftime("%m-%d-(%H-%M-%S)") + str(args.device) + '-' + args.model_type+args.model_size +'-'+ args.dim_to_iterate+'-'+str(args.n_trials)+'trials'+args.tfwriter_dir_append
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, dt_string))

    print('Device: ', args.device)

    # prep model dimensions
    if args.dim_to_iterate == 'dff':
        min_dim=1
        max_dim=768*4
        dims_to_test = list(range(args.dim_step, max_dim+1, args.dim_step)) # can fill in gap from dim=1 to dim=dim_step in post 
        # dims_to_test = range(max_dim, max_dim+1) # testing full model
    elif args.dim_to_iterate == 'heads':
        min_dim=1
        max_dim=12
        dims_to_test = list(range(max_dim, min_dim-1, -1))
        print('head dims to test: ', list(dims_to_test))
    print(dims_to_test)
    # prep inputs
    hidden_size=768
    inputs = {"input_ids": torch.ones(size=(args.per_gpu_eval_batch_size,args.max_seq_length), dtype=torch.long).to(args.device),
     "attention_mask": torch.ones(size=(args.per_gpu_eval_batch_size,args.max_seq_length),dtype=torch.long).to(args.device), 
     "token_type_ids":torch.zeros(size=(args.per_gpu_eval_batch_size,args.max_seq_length), dtype=torch.long).to(args.device), "labels": torch.zeros(size=(args.per_gpu_eval_batch_size,),dtype=torch.long).to(args.device)}
    hidden_states = torch.zeros(size=(args.per_gpu_eval_batch_size, args.max_seq_length, hidden_size), dtype=torch.float).to(args.device) # for bert layer 
    attn_mask = torch.ones(args.per_gpu_eval_batch_size, args.max_seq_length).to(args.device)
    # initial warmup with BERT base
    print('Starting ', args.init_warmup, ' second warmup')
    model = model_layer(config=config).to(args.device)
    start = time.time()
    while time.time()-start < args.init_warmup:
        model(hidden_states, attn_mask)
    print('done warmup')

    # latency experiment iterating through dimensions of intermediate width or number of heads
    use_layer = args.model_size == 'layer'
    if use_layer:
        input_tuple = (hidden_states.detach(),attn_mask)
    else:
        if args.model_type == 'bert':
            input_tuple = tuple([v for v in inputs.values()][:3])
        if args.model_type == 'distilbert':
            input_tuple = tuple([v for v in inputs.values()][:2])
            print(input_tuple)

    latencies = np.zeros((len(dims_to_test), args.n_trials), dtype=float)
    for i, dim in enumerate(tqdm(dims_to_test, desc="profiling")):
        if args.dim_to_iterate == 'dff':
            setattr(config, ffn_dim_name, dim)
            # config.intermediate_size = dim # this works for BERT config but not distilbert config. need to access hidden dim there
        model = model_layer(config=config).to(args.device) if use_layer else model_class(config=config).to(args.device) # uses tuple of dims
        if args.dim_to_iterate == 'heads':
            if dim < config.num_attention_heads:
                num_heads_to_prune = config.num_attention_heads - dim
                if use_layer:
                    model.attention.prune_heads(list(range(num_heads_to_prune)))
                    print('dim: ', dim, '  Pruned heads: ', model.attention.pruned_heads)
                else:
                    # model_heads = dict((layer, list(range(num_heads_to_prune))) for layer in range(config.num_hidden_layers))
                    model_heads = dict((layer, list(range(num_heads_to_prune))) for layer in range(config.num_hidden_layers))
                    model.prune_heads(model_heads)
                    print('dim: ', dim, '  Pruned heads: ', model.bert.encoder.layer[0].attention.pruned_heads)
        latencies[i] = eval_model(args, config, model, input_tuple) 
    print('calc stats')
    means = np.mean(latencies, axis=1)
    stds = np.std(latencies, axis=1)    
    mins = np.min(latencies, axis=1)
    maxes = np.max(latencies, axis=1)
    csv_data = np.array([np.array(dims_to_test), means, stds, mins, maxes]).T
    save_file = os.path.join(args.output_dir, dt_string,'data.csv')
    print(save_file)
    np.savetxt(save_file, csv_data, '%.5E', header='dim,mean,std,min,max', delimiter=',')

    if args.dim_step > 1: # need to interpolate values with linear interpolation
        interp_data = []
        full_dims = list(range(1,max_dim+1))
        for data in [means, stds, mins, maxes]:
            interp_data.append(np.interp(full_dims, dims_to_test, data, left=data[0]))
        csv_data_interp = np.array([np.array(full_dims),]+interp_data).T
        save_file = os.path.join(args.output_dir, dt_string,'data_interp.csv')
        print(save_file)
        np.savetxt(save_file, csv_data_interp, '%.5E', header='dim,mean,std,min,max', delimiter=',')
        for i in range(len(full_dims)):
            writer.add_scalar('Mean interpolated Latency', interp_data[0][i], full_dims[i])

    # print to tensorboard
    print('printing to tensorboard')
    for i in range(len(means)):
        writer.add_scalar('Mean Observed Latency', means[i], dims_to_test[i])
        writer.add_scalar('STD Observed Latency', stds[i], dims_to_test[i])
        writer.add_scalar('Min Observed Latency', mins[i], dims_to_test[i])
        writer.add_scalar('Max Observed Latency', maxes[i], dims_to_test[i])
    
    writer.close()

# this function is model invariant
def eval_model(args, config, model, input_tuple):
    model.eval()
    with torch.no_grad():
        model_traced = torch.jit.trace(model, input_tuple) #BUG this line causing memory leak. Commenting out removes mem leak. creates new model every time and doesn't get cleared
        # model_traced = model
        for j in range(args.n_warmup):
            out = model_traced(*input_tuple)
        # moved this back under torch.no_grad dec 30 653
        latencies = [0]*args.n_trials
        for trial in range(args.n_trials):
            if args.device.type=='cuda':
                torch.cuda.synchronize(args.device)
            then = time.time()
            out = model_traced(*input_tuple)
            if args.device.type=='cuda':
                torch.cuda.synchronize(args.device)
            now = time.time()
            latencies[trial] = now-then
    return latencies


#  def build_shape_dict(name: str, tensor, is_input: bool, seq_len: int):
#     if isinstance(tensor, (tuple, list)):
#         return [build_shape_dict(name, t, is_input, seq_len) for t in tensor]

#     else:
#         # Let's assume batch is the first axis with only 1 element (~~ might not be always true ...)
#         axes = {[axis for axis, numel in enumerate(tensor.shape) if numel == 1][0]: "batch"}
#         if is_input:
#             if len(tensor.shape) == 2:
#                 axes[1] = "sequence"
#             else:
#                 raise ValueError(f"Unable to infer tensor axes ({len(tensor.shape)})")
#         else:
#             seq_axes = [dim for dim, shape in enumerate(tensor.shape) if shape == seq_len]
#             axes.update({dim: "sequence" for dim in seq_axes})

#     # print(f"Found {'input' if is_input else 'output'} {name} with shape: {axes}")
#     return axes


# this is in progress
# from torch.onnx import export, load, backend
# def convert_to_trt(model, input_tuple):
#     '''convert pt model to trt'''
#     # convert to onnx
#     outputs=model(*input_tuple)
#     outputs_flat = []
#     for output in outputs:
#         if isinstance(output, (tuple, list)):
#             outputs_flat.extend(output)
#         else:
#             outputs_flat.append(output)
#     print('getting names')
#     output_names = [f"output_{i}" for i in range(len(outputs_flat))]
#     print('getting axes')
#     output_dynamic_axes = {k: build_shape_dict(k, v, False, seq_len) for k, v in zip(output_names, outputs_flat)}
#     dynamic_axes = {'input':{0:'batch'}, 'output':{0:'batch'}}
#     print('starting export')
#     export(
#         model=model, 
#         args=input_tuple, 
#         f='/home/meil-ah/ahoffman/neuron_pruning/struct-pruning/bert_profiler/onnx_cache/model.onnx', 
#         input_names=["input_ids", "attention_mask", "token_type_ids"],
#         output_names=output_names,
#         dynamic_axes=dynamic_axes,
#         do_constant_folding=True,
#         use_external_data_format=False,
#         enable_onnx_checker=True,
#         opset_version=11
#     )

#     # convert onnx to trt
#     model = load("/home/meil-ah/ahoffman/neuron_pruning/struct-pruning/bert_profiler/onnx_cache/model.onnx")
#     engine = backend.prepare(model, device='cuda')
#     output_data = engine.run(input_tuple)
#     print(output_data)
#     print(output_data[0])


# def eval_trt_model(args, config, model_class, use_layer, input_tuple):
#     '''convert model to tensorrt and evaluate it'''
#     model = BertLayer(config=config).to(args.device) if use_layer else model_class(config=config).to(args.device) # uses tuple of dims
#     if args.dim_to_iterate == 'heads':
#         if dim < config.num_attention_heads:
#             num_heads_to_prune = config.num_attention_heads - dim
#             if use_layer:
#                 model.attention.prune_heads(list(range(num_heads_to_prune)))
#                 print('dim: ', dim, '  Pruned heads: ', model.attention.pruned_heads)
#             else:
#                 # model_heads = dict((layer, list(range(num_heads_to_prune))) for layer in range(config.num_hidden_layers))
#                 model_heads = dict((1,list(range(num_heads_to_prune)),)) # just pruning from second layer
#                 model.prune_heads(model_heads)
#                 print('dim: ', dim, '  Pruned heads: ', model.bert.encoder.layer[0].attention.pruned_heads)
#     model.eval()
#     with torch.no_grad():
#         # convert model
#         traced_model = convert_to_trt(model, input_tuple)
#         latencies = [0]*args.n_trials
#         for trial in range(args.n_trials):
#             with torch.no_grad():
#                 then = time.time()
#                 out = model_traced(*input_tuple)
#                 # out=model(*input_tuple)
#                 now = time.time()
#                 latencies[trial] = now-then
#     return latencies

if __name__=='__main__':
    # cprof_main()
    latency_measurement()

    # STATSFILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),'eval_stats.csv')
    # cProfile.run('main()', STATSFILE)
    # process_stats(STATSFILE)

#python3 /home/ahoffman/research/neuron_pruning/struct-pruning/bert_profiler/bert_profile.py \
#--output_dir /home/ahoffman/research/neuron_pruning/struct-pruning/bert_profiler/latency_viz \
#--n_trials 100 --n_warmup 5 --init_warmup 40 --dim_to_iterate dff --model_size layer --tfwriter_dir_append 1080ti
