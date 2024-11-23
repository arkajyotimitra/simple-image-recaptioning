# simple-image-recaptioning

Recaption large (Web)Datasets with [`vllm`](https://github.com/vllm-project/vllm/) and save the artifacts. It is NOT a library. It, instead, provides reference points that you're free to use and modify. 

You can use `LLaVA` or `Qwen-VL` to "re-caption" your images.

> [!NOTE]
> I use the code of this repository for my projects and I don't claim this project to be out of the world. If you want to contribute an enhancement feature, you're more than welcome to open a PR. I'd greatly appreciate it.

## Getting started

Install the requirements: `pip install -r requirements.txt`.

Then run:

```bash
python main.py \
    --data_path="https://huggingface.co/datasets/pixparse/cc3m-wds/resolve/main/cc3m-train-0000.tar"
```

This will recaption a single shard of the CC3M dataset and will serialize the artifacts inside a directory called `sample_outputs`. This directory will have:

* The original image with its hash as its filename.
* A JSON file with the same hash as the filename containing the original and predicted captions.

If you want to use multiple shards then do:

```bash
# full CC3M training set
python main.py \
    --data_path="pipe:curl -s -f -L https://huggingface.co/datasets/pixparse/cc3m-wds/resolve/main/cc3m-train-{0000..0575}.tar"
```

You can allow watermark detection by passing `--detect_watermarks`. Note that this will require the following things:

* `onnx` and `onnxruntime` dependencies.
* Install `pip install git+https://github.com/sayakpaul/watermark-detection`. Then follow [the steps](https://github.com/sayakpaul/watermark-detection?tab=readme-ov-file#onnx-usage-limited-to-convnext-tiny) to obtain the ONNX model needed for watermark detection.

By default, the script will use all the available GPUs. I tested the above commands on two A100s and on eight H100s.

## List of CLI Arguments

```shell
NAME
    main.py

SYNOPSIS
    main.py DATA_PATH <flags>

POSITIONAL ARGUMENTS
    DATA_PATH
        Type: str

FLAGS
    --model_name=MODEL_NAME
        Type: str
        Default: 'llava'
    -b, --batch_size=BATCH_SIZE
        Type: int
        Default: 48
    --dataloader_num_workers=DATALOADER_NUM_WORKERS
        Type: int
        Default: 8
    -o, --output_dir=OUTPUT_DIR
        Type: str
        Default: 'sample_outputs'
    --max_tokens=MAX_TOKENS
        Type: int
        Default: 120
    --detect_watermarks=DETECT_WATERMARKS
        Type: typing.Union[bool, str]
        Default: False
```

## Principles

1. Recaptioning large image datasets has become a da-facto standard for the image generation community. So, I wanted to have a simple-yet-performant utility that would allow me to recaption large image datasets like [CC3M](https://huggingface.co/datasets/pixparse/cc3m-wds). This is why, I chose `vllm` as it provides optimized inference across multiple GPUs off-the-shelf.

2. [`webdataset`](https://github.com/webdataset/webdataset) is a common format used by practitioners to conduct training on large-scale datasets. So, I chose that as an entrypoint. Specifically, I assume that your image-caption pair dataset is already sharded into multiple `webdataset` archives. Refer [here](https://huggingface.co/datasets/pixparse/cc3m-wds) as an example. 

3. I need to be able to use multiple GPUs, overlapping communication and computation. But this project also works with a single GPU.

4. There has to be artifact serialization. This project serializes the original image, original caption, and the predicted caption in separate threads, not blocking the GPU(s).

5. There has to be watermark detection in the data curation pipeline at minimum. Otherwise, it messes up with the generation quality. In this project, it happens _during_ dataloading. To not clog the processes, we make use of ONNX for fast CPU-based inferencing.

6. Failures can happen during the captioning process so we need to able to avoid duplication. I have added a simple [`ExistsFilter`](https://github.com/sayakpaul/simple-image-recaptioning/blob/c150ce937cd371930eb67bb0fedac754757e7b23/data_processing.py#L56) filter to filter out the existing images that were serialized before interruptions.

## Code organization and modification

Ultimately, you'd want to modify the codebase to suit your needs. 

```bash
.
├── config.py -- specifies the prompt to be used to generate the captions and model id.
├── data_processing.py -- webdataset loading and processing code including watermark detection and caching.
├── main.py -- main entrypoint.
├── model.py -- loads the vllm engine and houses the simple inference function.
└── utils.py -- misc utilities.
```

If you have anything to modify, feel free to go into these files and modify them as per your needs. 

## Handy tips

* A good chunk of data processing used `PIL`. Simply replace your `Pillow` installation to use [`Pillow-SIMD`](https://github.com/uploadcare/pillow-simd) for better speed.
* Use `hf_transfer` for faster downloads from the Hugging Face Hub. Refer [here](https://huggingface.co/docs/huggingface_hub/main/en/package_reference/environment_variables#hfhubenablehftransfer) to know more. 

## Known limitations

Would really appreciate some contributions too :-)

* I believe the cache management system could be a bit optimized. It's likely IO bound, currently. Relevant links: [1](https://github.com/sayakpaul/simple-image-recaptioning/blob/c150ce937cd371930eb67bb0fedac754757e7b23/data_processing.py#L56), [2](https://github.com/sayakpaul/simple-image-recaptioning/blob/c150ce937cd371930eb67bb0fedac754757e7b23/data_processing.py#L99). 
* Better placement of the watermark detection module. I placed it inside the data pipeline because I wasn't sure how to make gel well with `vllm`. But this restricts higher throughputs a bit. 
* Be aware of [this issue](https://github.com/vllm-project/vllm/issues/8421) and [this fix](https://github.com/vllm-project/vllm/pull/8496) in `vllm`. 

## Acknowledgements

* Thanks to `vllm` for the amazing project.
* Thanks to `webdataset` for scalability.
* Thanks to Claude for pairing.
* Thanks to Hugging Face for letting me explore this and providing GPUs.
