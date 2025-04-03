## Pre-trained models w/ HuggingFace
```py
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.config 
```

## Device Placement with Accelerator 
```py
from accelerate import Accelerator 

accelerator = Accelerator() 

# Prepare objects for distributed training
# Arg order must match definition order
model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    model, 
    optimizer,
    train_dataloader,
    lr_scheduler
)

model.device # sanity check
```


## Training loop w/ Accelerator 
```py 
# Automatically move to device, change loss.backward() step
for inputs, targets in dataloader: 
    optimizer.zero_grad() 
    outputs = model(inputs) 

    loss = outputs.loss 
    accelerator.backward(loss)

    optimizer.step()
    scheduler.step() 

```

## Create Checkpoints 
```py
dataloader = DataLoader(train_dataset, batch_size=32)
dataloader = accelerator.prepare(dataloader)

checkpoint_dir = Path("preprocess_checkpoint")
accelerator.save_state(checkpoint_dir)

# Resume training later
accelerator.load_state(checkpoint_dir) 
```

## Gradient Accumulation w/ Accelerator 

Training with large batch sizes makes more robust gradient learning for quicker learning, but memory-intensive.

Gradient accumulation addresses this by summing gradietns over smaller batches. 

> Take large batch. Split into two. Scale loss (e.g., loss / 2). Compute backward pass for each new batch. Sum up new gradients. Update model

```py 
accelerator = Accelerator(gradient_accumulation_steps=2)

for index, (inputs, targets) in enumerate(dataloader): 
    with accelerator.accumulate(model): 
        outputs = model(inputs, labels=targets)

        loss = outputs.loss 
        accelerator.backward(loss)
 
        optimizer.step()
        lr_scheduler.step() 
        optimizer.zero_grad() 
```

## Mixed Precision 

It's easier to make computations with `FP16` than `FP32`; however, exclusively using `FP16` reduces accuracy. Mixed precision training is a method to speed up training and reducing memory usage, all while keeping critical values in FP32 to prevent numerical instability. 

> Compute fpass in `FP16`. Compute loss in `FP32`. Multiply by scale factor to avoid underflow. Compute bpass in `FP16`. Scale gradietns to undo the scaling operation.  Update model parameters in `FP32`. Store model parameters in `FP16`. Repeat

```py
# Enable mixed precision 
accelerator = Accelerator(mixed_precision="fp16")
```

# Optimizers 

> Adam has high memory usage, and has high speed. It uses adaptive learning rates and momentum.  
> AdamW has similar memory as Adam, and is slightly faster than AdamW. Weight decay is decoupled from the learning rate, better generalization
> Adagrad has very high memory usage (stores squared gradients for all parameters), and becomes slower over time. Has adaptive learning rates per parameter, but aggressively decays over time 
> 8-bit Adam has low memory (int8/16 instead of fp32), has low precision, is faster than Adam, and significantly reduces memory while maintaining performance. 



