#export LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH
import os

model_dir = "/tmp"
# os.system(f"wget https://huggingface.co/stabilityai/stable-diffusion-2-1/resolve/main/v2-1_768-ema-pruned.safetensors -P {model_dir}")
v2 = True
v_parameterization = True
project_name = "mecha_test"
pretrained_model_name_or_path = f"{model_dir}/v2-1_768-ema-pruned.safetensors" 
vae = "" 
train_data_dir = "/app/train_data/tmp"
in_json = "/app/train_data/tmp/meta_lat.json"
output_dir = "/app/sd_output/mecha_test"
resume_path = ""

# Check if directory exists
if not os.path.exists(output_dir):
  # Create directory if it doesn't exist
  os.makedirs(output_dir)

#V2 Inference
inference_url = "https://raw.githubusercontent.com/Stability-AI/stablediffusion/main/configs/stable-diffusion/"

if v2 and not v_parameterization:
  inference_url += "v2-inference.yaml"
if v2 and v_parameterization:
  inference_url += "v2-inference-v.yaml"

try:
  if v2 and not os.path.isfile(f"{output_dir}/{project_name}.yaml"):
    os.system(f"wget {inference_url} -O {output_dir}/{project_name}.yaml")
    print("File successfully downloaded")
except:
  print("There was an error downloading the file. Please check the URL and try again.")
  
from prettytable import PrettyTable
import textwrap
import yaml

#@title ## 5.2. Start Fine-Tuning
#@markdown ### Define Parameter
train_batch_size = 24 #@param {type:"number"}
train_text_encoder = True #@param {'type':'boolean'}
max_train_steps = 5000 #@param {type:"number"}
mixed_precision = "fp16" #@param ["no","fp16","bf16"] {allow-input: false}
save_precision = "fp16" #@param ["float", "fp16", "bf16"] {allow-input: false}
save_n_epochs_type = "save_every_n_epochs" #@param ["save_every_n_epochs", "save_n_epoch_ratio"] {allow-input: false}
save_n_epochs_type_value = 1 #@param {type:"number"}
save_model_as = "safetensors" #@param ["ckpt", "safetensors", "diffusers", "diffusers_safetensors"] {allow-input: false}
resolution = 768 #@param {type:"number"}
max_token_length = 225 #@param {type:"number"}
clip_skip = 2 #@param {type:"number"}
learning_rate = 2e-6 #@param {type:"number"}
lr_scheduler = "constant" #@param  ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"] {allow-input: false}
dataset_repeats = 1 #@param {type:"number"}
use_8bit_adam = True #@param {type:"boolean"}
gradient_checkpointing = True #@param {type:"boolean"}
gradient_accumulation_steps = 1 #@param {type:"number"}
seed = 0 #@param {type:"number"}
logging_dir = "/tf_logs"
log_prefix = project_name
additional_argument = "--save_state --xformers" #@param {type:"string"}
print_hyperparameter = True #@param {type:"boolean"}

train_command=f"""
accelerate launch --num_cpu_threads_per_process=8 fine_tune.py \
  {"--v2" if v2 else ""} \
  {"--v_parameterization" if v2 and v_parameterization else ""} \
  --pretrained_model_name_or_path={pretrained_model_name_or_path} \
  {"--vae=" + vae if vae else ""} \
  --train_data_dir={train_data_dir} \
  --in_json={in_json} \
  --output_dir={output_dir} \
  {"--resume=" + resume_path if resume_path else ""} \
  {"--output_name=" + project_name if project_name else ""} \
  --mixed_precision={mixed_precision} \
  --save_precision={save_precision} \
  {"--save_every_n_epochs=" + format(save_n_epochs_type_value) if save_n_epochs_type=="save_every_n_epochs" else ""} \
  {"--save_n_epoch_ratio=" + format(save_n_epochs_type_value) if save_n_epochs_type=="save_n_epoch_ratio" else ""} \
  --save_model_as={save_model_as} \
  --resolution={resolution} \
  --train_batch_size={train_batch_size} \
  --max_token_length={max_token_length} \
  {"--train_text_encoder" if train_text_encoder else ""} \
  {"--use_8bit_adam" if use_8bit_adam else ""} \
  --learning_rate={learning_rate} \
  --dataset_repeats={dataset_repeats} \
  --max_train_steps={max_train_steps} \
  {"--seed=" + format(seed) if seed > 0 else ""} \
  {"--gradient_checkpointing" if gradient_checkpointing else ""} \
  {"--gradient_accumulation_steps=" + format(gradient_accumulation_steps) } \
  {"--clip_skip=" + format(clip_skip) if v2 == False else ""} \
  --logging_dir={logging_dir} \
  --log_prefix={log_prefix} \
  {additional_argument}
  """
debug_params = ["v2", \
                "v_parameterization", \
                "pretrained_model_name_or_path", \
                "vae", \
                "train_data_dir", \
                "in_json", \
                "output_dir", \
                "resume_path", \
                "project_name", \
                "mixed_precision", \
                "save_precision", \
                "save_n_epochs_type", \
                "save_n_epochs_type_value", \
                "save_model_as", \
                "resolution", \
                "train_batch_size", \
                "max_token_length", \
                "use_8bit_adam", \
                "learning_rate", \
                "lr_scheduler", \
                "dataset_repeats", \
                "train_text_encoder", \
                "max_train_steps", \
                "seed", \
                "gradient_checkpointing", \
                "gradient_accumulation_steps", \
                "clip_skip", \
                "logging_dir", \
                "log_prefix", \
                "additional_argument"]

if print_hyperparameter:
    table = PrettyTable()
    table.field_names = ["Hyperparameter", "Value"]
    for params in debug_params:
        if params != "":
            if globals()[params] == "":
                value = "False"
            else:
                value = globals()[params]
            table.add_row([params, value])
    table.align = "l"
    print(table)

    arg_list = train_command.split()
    mod_train_command = {'command': arg_list}
    
    train_folder = os.path.dirname(output_dir)
    
    # save the YAML string to a file
    with open(str(train_folder)+'/finetune_cmd.yaml', 'w') as f:
        yaml.dump(mod_train_command, f)

f = open("./train.sh", "w")
f.write(train_command)
f.close()