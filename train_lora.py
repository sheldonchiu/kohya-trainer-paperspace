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
  if v2:
    os.system(f"wget {inference_url} -O {output_dir}/{project_name}.yaml")
    print("File successfully downloaded")
except:
  print("There was an error downloading the file. Please check the URL and try again.")

#@title ## 5.2. Define Specific LoRA Training Parameters

#@markdown ## LoRA - Low Rank Adaptation Fine-tuning

#@markdown Some people recommend setting the `network_dim` to a higher value.
network_dim = 128 #@param {'type':'number'}
#@markdown For weight scaling in LoRA, it is better to set `network_alpha` the same as `network_dim` unless you know what you're doing. A lower `network_alpha` requires a higher learning rate. For example, if `network_alpha = 1`, then `unet_lr = 1e-3`.
network_alpha = 128 #@param {'type':'number'}
network_module = "networks.lora"

#@markdown `network_weights` can be specified to resume training.
network_weights = "" #@param {'type':'string'}

#@markdown By default, both Text Encoder and U-Net LoRA modules are enabled. Use `network_train_on` to specify which module to train.
network_train_on = "both" #@param ['both','unet_only', 'text_encoder_only'] {'type':'string'}

#@markdown It is recommended to set the `text_encoder_lr` to a lower learning rate, such as `5e-5`, or to set `text_encoder_lr = 1/2 * unet_lr`.
unet_lr = 1e-4 #@param {'type':'number'}
text_encoder_lr = 5e-5 #@param {'type':'number'}
lr_scheduler = "constant" #@param ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"] {allow-input: false}

#@markdown If `lr_scheduler = cosine_with_restarts`, update `lr_scheduler_num_cycles`.
lr_scheduler_num_cycles = 1 #@param {'type':'number'}
#@markdown If `lr_scheduler = polynomial`, update `lr_scheduler_power`.
lr_scheduler_power = 1 #@param {'type':'number'}

#@markdown Check the box to not save metadata in the output model.
no_metadata = False #@param {type:"boolean"}
training_comment = "this comment will be stored in the metadata" #@param {'type':'string'}

print("Loading network module:", network_module)
print(f"{network_module} dim set to:", network_dim)
print(f"{network_module} alpha set to:", network_alpha)

if network_weights == "":
  print("No LoRA weight loaded.")
else:
  if os.path.exists(network_weights):
    print("Loading LoRA weight:", network_weights)
  else:
    print(f"{network_weights} does not exist.")
    network_weights =""

if network_train_on == "unet_only":
  print("Enabling LoRA for U-Net.")
  print("Disabling LoRA for Text Encoder.")

if network_train_on == "unet_only":
  print("Enable LoRA for U-Net")
  print("Disable LoRA for Text Encoder")
  print("UNet learning rate: ", unet_lr)
if network_train_on == "text_encoder_only":
  print("Disabling LoRA for U-Net")
  print("Enabling LoRA for Text Encoder")
  print("Text encoder learning rate: ", text_encoder_lr)
else:
  print("Enabling LoRA for U-Net")
  print("Enabling LoRA for Text Encoder")
  print("UNet learning rate: ", unet_lr)
  print("Text encoder learning rate: ", text_encoder_lr)

print("Learning rate Scheduler:", lr_scheduler)

if lr_scheduler == "cosine_with_restarts":
  print("- Number of cycles: ", lr_scheduler_num_cycles)
elif lr_scheduler == "polynomial":
  print("- Power: ", lr_scheduler_power)

# Printing the training comment if metadata is not disabled and a comment is present
if not no_metadata:
  if training_comment: 
    training_comment = training_comment.replace(" ", "_")
    print("Training comment:", training_comment)
else:
  print("Metadata won't be saved")
  
from prettytable import PrettyTable
import textwrap
import yaml

#@title ## 5.2. Start Fine-Tuning
#@markdown ### Define Parameter
train_batch_size = 4 #@param {type:"number"}
num_epochs = 10 #@param {type:"number"}
dataset_repeats = 1 #@param {type:"number"}
mixed_precision = "fp16" #@param ["no","fp16","bf16"] {allow-input: false}
save_precision = "fp16" #@param ["float", "fp16", "bf16"] {allow-input: false}
save_n_epochs_type = "save_n_epoch_ratio" #@param ["save_every_n_epochs", "save_n_epoch_ratio"] {allow-input: false}
save_n_epochs_type_value = 1 #@param {type:"number"}
save_model_as = "safetensors" #@param ["ckpt", "pt", "safetensors"] {allow-input: false}
resolution = 768 #@param {type:"number"}
max_token_length = 225 #@param {type:"number"}
clip_skip = 2 #@param {type:"number"}
use_8bit_adam = True #@param {type:"boolean"}
gradient_checkpointing = True #@param {type:"boolean"}
gradient_accumulation_steps = 1 #@param {type:"number"}
seed = 0 #@param {type:"number"}
logging_dir = "/app/sd_output/logs"
log_prefix = project_name
additional_argument = "--xformers" #@param {type:"string"}
print_hyperparameter = True #@param {type:"boolean"}
flip = False

train_command=f"""
accelerate launch --num_cpu_threads_per_process=8 train_network.py \
  {"--v2" if v2 else ""} \
  {"--v_parameterization" if v2 and v_parameterization else ""} \
  --network_dim={network_dim} \
  --network_alpha={network_alpha} \
  --network_module={network_module} \
  {"--network_weights=" + network_weights if network_weights else ""} \
  {"--network_train_unet_only" if network_train_on == "unet_only" else ""} \
  {"--network_train_text_encoder_only" if network_train_on == "text_encoder_only" else ""} \
  {"--unet_lr=" + format(unet_lr) if unet_lr else ""} \
  {"--text_encoder_lr=" + format(text_encoder_lr) if text_encoder_lr else ""} \
  {"--no_metadata" if no_metadata else ""} \
  {"--training_comment=" + training_comment if training_comment and not no_metadata else ""} \
  --lr_scheduler={lr_scheduler} \
  {"--lr_scheduler_num_cycles=" + format(lr_scheduler_num_cycles) if lr_scheduler == "cosine_with_restarts" else ""} \
  {"--lr_scheduler_power=" + format(lr_scheduler_power) if lr_scheduler == "polynomial" else ""} \
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
  {"--use_8bit_adam" if use_8bit_adam else ""} \
  --dataset_repeats={dataset_repeats} \
  --max_train_epochs={num_epochs} \
  {"--seed=" + format(seed) if seed > 0 else ""} \
  {"--gradient_checkpointing" if gradient_checkpointing else ""} \
  {"--gradient_accumulation_steps=" + format(gradient_accumulation_steps) } \
  {"--clip_skip=" + format(clip_skip) if v2 == False else ""} \
  --logging_dir={logging_dir} \
  --log_prefix={log_prefix} \
  {"--flip_aug" if flip else ""} \
  {additional_argument}
  """

debug_params = ["v2", \
                "v_parameterization", \
                "network_dim", \
                "network_alpha", \
                "network_module", \
                "network_weights", \
                "network_train_on", \
                "unet_lr", \
                "text_encoder_lr", \
                "no_metadata", \
                "training_comment", \
                "lr_scheduler", \
                "lr_scheduler_num_cycles", \
                "lr_scheduler_power", \
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
                "dataset_repeats", \
                "num_epochs", \
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
    with open(str(train_folder)+'/finetune_lora_cmd.yaml', 'w') as f:
        yaml.dump(mod_train_command, f)

with open("./train.sh", "w") as f:
    f.write(train_command)

# !chmod +x ./train.sh
# !./train.sh