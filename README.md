# pFedSCPT

**Selective Class Prototype Transfer with Persistent Memory for Personalized Federated Learning**

This is a self-contained core implementation of pFedSCPT. Clients collaborate through class prototypes and a persistent server memory. Each client keeps a private feature adapter, classifier, and EMA teacher. Only encoder parameters are averaged; batch normalization running statistics remain local.

## Files

Only two repository files are required:

~~~text
main.py
README.md
~~~

**main.py** includes model definitions, dataset access, client partitioning, CEMA memory, PRKI training, and evaluation. It does not import Python files from the original project. Datasets and generated run outputs are stored separately.

## Environment

Validated with Python 3.9.25, PyTorch 2.8.0, torchvision 0.23.0, and NumPy 2.0.2. Pillow is also required.

~~~bash
python -m pip install torch==2.8.0 torchvision==0.23.0 numpy==2.0.2 pillow
python main.py --help
~~~

For GPU training, use a PyTorch installation compatible with your CUDA environment. The default device setting, **auto**, selects CUDA when available and otherwise uses the CPU. Use **--device cuda:0** or **--device cpu** to select a device explicitly.

## Data

### CIFAR-10 and CIFAR-100

The program downloads missing CIFAR data through torchvision to **--datadir**. An existing torchvision-format dataset directory can also be used.

~~~text
data/
  cifar-10-batches-py/       # CIFAR-10, if used
  cifar-100-python/          # CIFAR-100, if used
~~~

### Tiny-ImageNet

Download and extract Tiny-ImageNet separately. **--datadir** can point to its parent directory or directly to **tiny-imagenet-200**. Both the official flat validation layout and validation images organized into matching class folders are supported.

~~~text
data/tiny-imagenet-200/
  train/<class_id>/images/*.JPEG
  val/images/*.JPEG
  val/val_annotations.txt
~~~

All images are processed at 32 x 32 pixels, following the existing implementation. The default backbone is SimpleCNN for CIFAR-10/CIFAR-100 and ResNet-18 with a 3 x 3, stride-1 stem for Tiny-ImageNet. Their feature dimensions are 256 and 512. Neither backbone uses pretrained weights.

### Client train/test protocol

The original training and test sources are pooled and randomly divided at a 3:1 ratio. These disjoint pools are allocated to clients using the same class-wise proportions. This follows the manuscript's pooled-data protocol; it does not retain the official dataset train/test boundary. Realized client proportions differ slightly because of integer allocation.

| Arguments | Partition |
| --- | --- |
| --partition dirichlet --beta 0.1 | Dirichlet label skew; beta is the paper's concentration alpha |
| --partition noniid-#label20 | Fixed-label skew with 20 assigned labels per client |
| --partition homo | Approximately uniform class allocation |

Dirichlet mode retains the original allocator's safeguard: it redistributes class mass to give selected class owners a one-sample expected floor and selects at least two owners when feasible. This adjusted allocation is not an unmodified Dirichlet draw. The source qualification threshold for CEMA is applied later and does not set this allocation floor.

The program rejects an empty client training or test set rather than substituting a sample. Fixed-label partitions require enough total label slots to cover every class. The seed controls partitioning and initialization. Allocated client indices are saved for each run.

## Run

Run these commands from the directory containing **main.py**.

CIFAR-100 with 20 clients, full participation, and Dirichlet alpha = 0.1:

~~~bash
python main.py --dataset cifar100 --partition dirichlet --beta 0.1 --n-parties 20 --comm-round 200 --seed 0 --datadir ./data --device auto --output-dir ./runs/cifar100_a01_seed0 --save-checkpoint
~~~

CIFAR-10 with two assigned labels per client:

~~~bash
python main.py --dataset cifar10 --partition noniid-#label2 --n-parties 20 --comm-round 200 --seed 0 --datadir ./data --output-dir ./runs/cifar10_label2_seed0
~~~

Tiny-ImageNet with the default ResNet-18:

~~~bash
python main.py --dataset tinyimagenet --partition dirichlet --beta 0.1 --n-parties 20 --comm-round 200 --seed 0 --datadir ./data --output-dir ./runs/tinyimagenet_a01_seed0
~~~

A short installation check using real CIFAR-100:

~~~bash
python main.py --dataset cifar100 --comm-round 2 --epochs 1 --e-distill 1 --device cpu --output-dir ./runs/installation_check
~~~

The installation check still needs the dataset and does not evaluate convergence. Use a new output directory for each run. Omitting **--output-dir** creates a timestamped directory automatically.

## Core method

The code has six numbered sections. The training helpers below are nested in **run_federated** where indicated.

| Component | Code location | Operation |
| --- | --- | --- |
| Private adaptation | FeatureAffineAdapter | Per-coordinate scale and bias, initialized to the identity |
| Client partitions | build_pfedscpt_paper_split | Disjoint pooled 3:1 split and matched class-wise client allocation |
| Source qualification | _build_cema_fixed_upload | Admit a class contribution only when its local count reaches the source threshold |
| Reliability tiers | _resolve_cema_tier | Assign strong, medium, or weak transfer weights |
| Persistent memory | _build_class_expert_memory, inside run_federated | Aggregate class prototypes/variances and update server memory |
| Personalized transfer | PRKI loop, inside run_federated | Real-image training, private teacher preservation, prototype alignment/contrast, and feature replay |

### CEMA: class-level expert memory aggregation

Clients compute normalized encoder-space class prototypes and diagonal feature variances from training images. A source class qualifies when its count reaches **local_qual_min_count** (default 7). Eligible contributions are weighted by the class count raised to **global_gamma**, multiplied by local class reliability. Local reliability is the count divided by **class_reliable_min**, capped at one. The optional quality factor is fixed to one.

Each class has a fixed upload slot containing weighted prototype/variance sums, the weight sum, eligible-client count, eligible sample count, and total sample count. Server reliability combines eligible-client support and eligible-sample coverage. Tiers control intervention:

- **strong:** full transfer weight;
- **medium:** reliability-capped transfer, limited by **medium_cema_max_weight**;
- **weak:** retained memory without prototype intervention.

Prototypes and variances use EMA. Existing entries persist when no new eligible contribution arrives. Historical support, reliability, and support-count fields use separate maxima. These fields do not count distinct contributors across rounds, and reliability is not recomputed from the independent maxima.

The aggregate-only interface is simulated by summation in a single process. This release does not implement cryptographic secure aggregation, differential privacy, or a distributed transport layer.

### PRKI: prototype replay knowledge injection

The private adapter maps shared encoder features and replayed prototype features into inputs for the private classifier. Its scale and bias receive identity regularization. It remains local during encoder averaging.

Replay covers classes that are observed and rare on the receiving client and have strong or medium server memory. Synthetic features are sampled around server prototypes using diagonal variances and then normalized. Replay updates only the adapter and classifier; it has no encoder computation path. The replay count is **per eligible class per real minibatch**. Very small observed classes receive a count boost.

Real-image training combines label CE, private EMA teacher preservation, shared/adapted prototype alignment, and strong-prototype contrast. Medium alignment targets are used only for observed rare classes. Locally missing classes receive no replay; strong missing-class prototypes can be contrastive negatives. All loss weights are exposed through **--help**.

### Round order

1. Round 0 performs local CE warm-up and initializes private frozen teachers. It does not average encoders at the end of that round.
2. Later rounds average selected clients' encoder parameters and broadcast them to all clients. BN running statistics remain local.
3. Selected clients extract CEMA statistics **after encoder replacement and before PRKI training**. The server updates memory from these statistics.
4. Each selected client's private teacher is refreshed by EMA. The client runs PRKI against the current memory and teacher.
5. Selected clients are evaluated on local test sets. Their trained encoder states are available for aggregation at the next round's start.

Round indices start at zero. Full participation is the default. With **--sample** below one, round 0 still warms up every client; later metrics cover only clients selected in that round.

## Key configuration

| Argument | Default | Meaning |
| --- | --- | --- |
| --n-parties | 20 | Client count |
| --comm-round | 200 | Total rounds, including warm-up |
| --sample | 1.0 | Participation fraction after warm-up |
| --batch-size | 100 | Local batch size |
| --epochs | 5 | Round-0 CE warm-up epochs |
| --e-distill | 7 | PRKI epochs per active round |
| --lr | auto | 0.005 for SimpleCNN; 0.1 for ResNet-18 |
| --rho | 0.9 | SGD momentum |
| --local-qual-min-count | 7 | Source class count threshold |
| --global-gamma | 0.1 | Class count weighting exponent |
| --class-reliable-min | 12 | Count for full local reliability |
| --global-min-support / --tau-B | 3 / 0.6 | Strong tier thresholds |
| --medium-min-support / --medium-tau-B | 2 / 0.3 | Medium tier thresholds |
| --medium-cema-max-weight | 0.5 | Medium transfer cap |
| --eta-B | 0.82 | Server memory EMA decay |
| --ema-decay | 0.97 | Private teacher EMA decay |
| --lambda-rep-seen | 0.2 | Replay loss weight |
| --lambda-align | 0.05 | Shared prototype alignment weight |
| --lambda-pcon | 0.02 | Strong prototype contrastive weight |
| --lambda-adapter-reg | 0.01 | Adapter identity regularization |
| --use-cema-memory | 1 | Set to 0 for a no-memory ablation |

The learning rate halves every 40 rounds. Multiword flags generally accept both hyphens and the original underscore spelling. Resolved model, device, and learning rate values are saved in the run configuration.

### Configuration provenance and reproduction scope

This is a core release, not a complete reproduction package for all manuscript tables. Transfer coefficients follow the existing CIFAR-100 full-method launch script. The source count threshold is 7; the client count and 200-round budget follow the current manuscript. Dataset-specific tuned configurations, baseline runners, three-seed result aggregation, and plotting scripts are outside this release.

The existing launch script uses **7 PRKI epochs**, whereas the manuscript currently states **5 local epochs per round**. Both phases are exposed separately here. The default retains the script's seven PRKI epochs; use **--e-distill 5** for five. This discrepancy needs resolution before claiming exact reproduction of reported results.

The extraction adds source dataset caching, official Tiny-ImageNet validation annotation support, CPU offloading between sequential clients, rejection of empty splits, and run artifact saving. The alignment target guard now includes the alignment loss's own coefficient, so a positive alignment weight still works when other alignment coefficients are zero. Original project files are unchanged.

## Outputs and metrics

~~~text
config.json          # Resolved arguments and library versions
partition.npz        # Per-client indices into pooled source ordering
training.log         # Per-client losses and round summaries
metrics.csv          # One row per round
client_metrics.csv   # Per-client accuracy and grouped correct/total counts
final.pt             # Optional: enabled by --save-checkpoint
~~~

**overall** is the equal-client mean of personalized local-test Top-1 accuracies. **head** and **rarep** pool correct predictions and test sample counts across evaluated clients. Accuracy values are fractions in [0, 1]. An undefined grouped accuracy has an empty CSV cell.

Head classes have positive counts at least **rare_head_ratio** times the largest local training class count. RareP classes have positive counts no greater than the maximum of **rare_abs_threshold** and **rare_rel_threshold** times the largest count. These analysis subsets can overlap and do not necessarily cover every observed class.

The optional final checkpoint stores client encoders, adapters, classifiers, private teachers, class memory, configuration, and round history. Automatic resume and optimizer-state restoration are not implemented.

## Validation

Checks cover command-line loading, parity with original backbone outputs and partition indices, source qualification, all memory tiers, EMA/retention, private BN buffers, and replay gradient ownership. Short three-client CPU runs on synthetic images exercise the full method, no-memory ablation, alignment-only activation, metric saving, and checkpoints. These validate execution and extraction behavior. Full dataset training and manuscript accuracy reproduction were not rerun for this release.

## Attribution and license

This code was extracted from the project's training implementation, which builds on **FedConcat**, the code for *Exploiting Label Skew in Federated Learning with Model Concatenation*. The upstream MIT copyright and permission notice are preserved in the header of **main.py**. The ResNet-18 implementation retains the source project's backbone structure.

Paper title: *pFedSCPT: Selective Class Prototype Transfer with Persistent Memory for Personalized Federated Learning*. Publication metadata will be added when available.
