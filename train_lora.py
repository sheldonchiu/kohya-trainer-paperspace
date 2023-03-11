#export LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH
import os

model_dir = "/hy-tmp"
# os.system(f"wget https://huggingface.co/stabilityai/stable-diffusion-2-1/resolve/main/v2-1_768-ema-pruned.safetensors -P {model_dir}")
v2 = False
v_parameterization = False
project_name = "mecha_v2_sd15_lora"
pretrained_model_name_or_path = f"{model_dir}/Counterfeit-V2.5.safetensors" 
vae = "" 
train_data_dir = "/hy-tmp/mecha_ofa_wd14vit_v2"
in_json = "/hy-tmp/meta_lat_v2_test.json"
output_dir = "/hy-tmp/mecha_v2_sd15_lora"
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

#@title ## 5.2. LoRA and Optimizer Config
#@markdown ### LoRA Config:<br>
#@markdown Some people recommend setting the `network_dim` to a higher value.
network_dim = 128 #@param {'type':'number'}
#@markdown For weight scaling in LoRA, it is better to set `network_alpha` the same as `network_dim` unless you know what you're doing. A lower `network_alpha` requires a higher learning rate. For example, if `network_alpha = 1`, then `unet_lr = 1e-3`.
network_alpha = 128 #@param {'type':'number'}
network_module = "networks.lora"
#@markdown `network_weight` can be specified to resume training.
network_weight = "" #@param {'type':'string'}
#@markdown By default, both Text Encoder and U-Net LoRA modules are enabled. Use `network_train_on` to specify which module to train.
network_train_on = "both" #@param ['both','unet_only', 'text_encoder_only'] {'type':'string'}
#@markdown ### <br>Optimizer Config:<br>
#@markdown `AdamW8bit` was the old `--use_8bit_adam` . Use `DAdaptation` if you find it hard to set optimizer hyperparameter right.
optimizer_type = "DAdaptation" #@param ["AdamW", "AdamW8bit", "Lion", "SGDNesterov", "SGDNesterov8bit", "DAdaptation", "AdaFactor"]
#@markdown Additional arguments for optimizer, e.g: `"decouple=True weight_decay=0.01 betas=0.9,0.999 ..."`.
#@markdown You probably need to specify `optimizer_args` for custom optimizer, like using `decouple=True weight_decay=0.6` for `DAdaptation`.
optimizer_args = "decouple=True weight_decay=0.6"#@param {'type':'string'}
#@markdown Set `unet_lr` to `1.0` if you use `DAdaptation` optimizer, as it's a [free learning rate](https://github.com/facebookresearch/dadaptation) algorithm. 
#@markdown However `text_encoder_lr = 1/2 * unet_lr` still applied, so you need to set `0.5` for `text_encoder_lr`.
#@markdown Also actually you don't need to specify `learning rate` value if both `unet_lr` and `text_encoder_lr` are defined.
unet_lr = 1.0 #@param {'type':'number'}
text_encoder_lr = 0.5 #@param {'type':'number'}
lr_scheduler = "constant_with_warmup" #@param ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"] {allow-input: false}
lr_warmup_steps = 200 #@param {'type':'number'}
#@markdown You can define `num_cycles` value for `cosine_with_restarts` or `power` value for `polynomial` in the field below.
lr_scheduler_args = 1 #@param {'type':'number'}

lora_and_optimizer_config = ["network_dim",
                              "network_alpha",
                              "network_module",
                              "network_weight",
                              "optimizer_type", 
                              "optimizer_args", 
                              "unet_lr", 
                              "text_encoder_lr", 
                              "lr_scheduler", 
                              "lr_warmup_steps", 
                              "lr_scheduler_args"]

# if optimizer_type == "Lion":
#   print("Installing Lion Pytorch...")
#   !pip -q install lion-pytorch
# if optimizer_type == "DAdaptation":
#   print("Installing DAdaptation...")
#   !pip -q install dadaptation

print("- LoRA Config:")
print("Loading network module:", network_module)
print(f"{network_module} dim set to:", network_dim)
print(f"{network_module} alpha set to:", network_alpha)
if not network_weight:
  print("No LoRA weight loaded.")
else:
  if os.path.exists(network_weight):
    print("Loading LoRA weight:", network_weight)
  else:
    print(f"{network_weight} does not exist.")
    network_weight =""
print("Enable LoRA for U-Net") if network_train_on == "both" or "unet_only" else ""
print("Enabling LoRA for Text Encoder") if network_train_on == "both" or "text_encoder_only" else ""
print("Disable LoRA for U-Net") if network_train_on == "text_encoder_only" else ""
print("Disable LoRA for Text Encoder") if  network_train_on == "unet_only" else ""

print("- Optimizer Config:")
print(f"Using {optimizer_type} as Optimizer")
if optimizer_args:
  print(f"Optimizer Args :", optimizer_args)
print("UNet learning rate: ", unet_lr)
print("Text encoder learning rate: ", text_encoder_lr)
print("Learning rate warmup steps: ", lr_warmup_steps)
print("Learning rate Scheduler:", lr_scheduler)
if lr_scheduler == "cosine_with_restarts":
  print("- lr_scheduler_num_cycles: ", lr_scheduler_args)
elif lr_scheduler == "polynomial":
  print("- lr_scheduler_power: ", lr_scheduler_args)
  
from prettytable import PrettyTable
import textwrap
import yaml

#@title ## 5.3. Start Training
#@markdown ### Dataset Config
#@markdown Specify this if you want to multiply your dataset count in a single epochs. Good for data balancing.
dataset_repeats = 1 #@param {type:"number"}
resolution = 768 #@param {type:"slider", min:512, max:1024, step:128}
#@markdown You can ignore some tags or captions chosen at random when training your model. Specify `caption_dropout_every_n_epochs` to dropout caption every n epochs.	
caption_dropout_rate = 0.1 #@param {type:"slider", min:0, max:1, step:0.05}
tag_dropout_rate = 0 #@param {type:"slider", min:0, max:1, step:0.05}	
caption_dropout_every_n_epochs = None #@param {type:"number"}

#@markdown ### <br> Training Config
lowram = False #@param {type:"boolean"}
#@markdown Read [Diffusion With Offset Noise](https://www.crosslabs.org//blog/diffusion-with-offset-noise), in short, you can control and easily generating darker or light images by offset the noise when fine-tuning the model. Set to `0` by default, recommended value: `0.1`
noise_offset = 0.1 #@param {type:"number"}
max_train_type = "max_train_epochs" #@param ["max_train_steps", "max_train_epochs"]
max_train_type_value = 10 #@param {type:"number"}
train_batch_size = 24 #@param {type:"number"}
mixed_precision = "fp16" #@param ["no","fp16","bf16"] {allow-input: false}
save_precision = "fp16" #@param ["float", "fp16", "bf16"] {allow-input: false}
save_n_epochs_type = "save_every_n_epochs" #@param ["save_every_n_epochs", "save_n_epoch_ratio"] {allow-input: false}
save_n_epochs_type_value = 1 #@param {type:"number"}
save_model_as = "safetensors" #@param ["ckpt", "pt", "safetensors"] {allow-input: false}
max_token_length = 225 #@param {type:"number"}
clip_skip = 2 #@param {type:"number"}
gradient_checkpointing = True #@param {type:"boolean"}
gradient_accumulation_steps = 2 #@param {type:"number"}
seed = -1 #@param {type:"number"}
logging_dir = "/tf_logs"
additional_argument = "--shuffle_caption --xformers" #@param {type:"string"}
flip_aug = True
print_hyperparameter = True #@param {type:"boolean"}

train_command=f"""
accelerate launch --num_cpu_threads_per_process=8 train_network.py \
  {"--v2" if v2 else ""} \
  {"--v_parameterization" if v2 and v_parameterization else ""} \
  {"--output_name=" + project_name if project_name else ""} \
  --pretrained_model_name_or_path={pretrained_model_name_or_path} \
  {"--vae=" + vae if vae else ""} \
  --train_data_dir={train_data_dir} \
  --in_json={in_json} \
  --output_dir={output_dir} \
  {"--resume=" + resume_path if resume_path else ""} \
  \
  --network_dim={network_dim} \
  --network_alpha={network_alpha} \
  --network_module={network_module} \
  {"--network_weights=" + network_weight if network_weight else ""} \
  {"--network_train_unet_only" if network_train_on == "unet_only" else ""} \
  {"--network_train_text_encoder_only" if network_train_on == "text_encoder_only" else ""} \
  --optimizer_type={optimizer_type} \
  {"--optimizer_args " + optimizer_args if optimizer_args else ""} \
  --learning_rate={unet_lr} \
  {"--unet_lr=" + format(unet_lr) if unet_lr else ""} \
  {"--text_encoder_lr=" + format(text_encoder_lr) if text_encoder_lr else ""} \
  --lr_scheduler={lr_scheduler} \
  {"--lr_warmup_steps=" + format(lr_warmup_steps) if lr_warmup_steps else ""} \
  {"--lr_scheduler_num_cycles=" + format(lr_scheduler_args) if lr_scheduler == "cosine_with_restarts" else ""} \
  {"--lr_scheduler_power=" + format(lr_scheduler_args) if lr_scheduler == "polynomial" else ""} \
  \
  --dataset_repeats={dataset_repeats} \
  --resolution={resolution} \
  {"--caption_dropout_rate=" + format(caption_dropout_rate) if caption_dropout_rate else ""} \
  {"--caption_tag_dropout_rate=" + format(tag_dropout_rate) if tag_dropout_rate else ""} \
  {"--caption_dropout_every_n_epochs=" + format(caption_dropout_every_n_epochs) if caption_dropout_every_n_epochs else ""} \
  \
  {"--lowram" if lowram else ""} \
  {"--noise_offset=" + format(noise_offset) if noise_offset > 0 else ""} \
  --train_batch_size={train_batch_size} \
  {"--max_train_epochs=" + format(max_train_type_value) if max_train_type == "max_train_epochs" else ""} \
  {"--max_train_steps=" + format(max_train_type_value) if max_train_type == "max_train_steps" else ""} \
  --mixed_precision={mixed_precision} \
  --save_precision={save_precision} \
  {"--save_every_n_epochs=" + format(save_n_epochs_type_value) if save_n_epochs_type=="save_every_n_epochs" else ""} \
  {"--save_n_epoch_ratio=" + format(save_n_epochs_type_value) if save_n_epochs_type=="save_n_epoch_ratio" else ""} \
  --save_model_as={save_model_as} \
  --max_token_length={max_token_length} \
  {"--seed=" + format(seed) if seed > 0 else ""} \
  {"--gradient_checkpointing" if gradient_checkpointing else ""} \
  {"--gradient_accumulation_steps=" + format(gradient_accumulation_steps) } \
  {"--clip_skip=" + format(clip_skip) if not v2 else ""} \
  --logging_dir={logging_dir} \
  --log_prefix={project_name} \
  {"--flip_aug" if flip_aug else ""} \
  {additional_argument}
  """
dataset_config = ["dataset_repeats",
                  "resolution",
                  "caption_dropout_rate",
                  "tag_dropout_rate",
                  "caption_dropout_every_n_epochs"]

training_config= ["lowram",
                  "noise_offset",
                  "max_train_type",
                  "max_train_type_value",
                  "train_batch_size",
                  "mixed_precision",
                  "save_precision",
                  "save_n_epochs_type",
                  "save_n_epochs_type_value",
                  "save_model_as",
                  "max_token_length",
                  "clip_skip",
                  "gradient_checkpointing",
                  "gradient_accumulation_steps",
                  "seed",
                  "logging_dir",
                  "additional_argument"]

arg_list = train_command.split()
mod_train_command = {'command': arg_list}
    
with open(os.path.join(output_dir,'finetune_lora_cmd.yaml'), 'w') as f:
  yaml.dump(mod_train_command, f)

with open("./train.sh", "w") as f:
    f.write(train_command)

# !chmod +x ./train.sh
# !./train.sh