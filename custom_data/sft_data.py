from datasets import load_dataset, concatenate_datasets
from .utils import make_masked_sft_collator
from .reasoning_datasets_info import (
    DATA_CONFIGS, wrap_string_between_tag, grab_text_between_tag, get_tags,
    ReasoningData)


def add_indices(ds):
    if "__index" not in ds.column_names:
        ds = ds.map(lambda x, i: {"__index": i}, with_indices=True)
    return ds


def get_process_line_fn(dataset_id_or_path):
    data: ReasoningData = DATA_CONFIGS[dataset_id_or_path]
    system_prompt = data.system_prompt

    def process_line_fn(line, tokenizer):
        question_content, thought_process_and_solution = (
            data.extract_question_and_completion_from_line(line))
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": question_content,
            },
            {
                "role": "assistant",
                "content": thought_process_and_solution,
            }
        ]
        line_text = tokenizer.apply_chat_template(
            messages, tokenize=False, continue_final_message=False)
        return {"text": line_text}
    return process_line_fn


def load_formatted_sft_dataset(
        tokenizer,
        dataset_id_or_path,
        dataset_local_directory=None,
        train_split='train',
        val_split=None,
        process_line_fn=None,
        model_name_or_path=None,
        completion_only_training=True,
        custom_start_of_response=None,
        keep_columns=None,
        add_dataset_indices=False,
        artificial_epochs=None,
        **dataset_loading_kwargs,
):

    if dataset_local_directory is None:
        dataset_local_directory = dataset_id_or_path
    dataset = load_dataset(dataset_local_directory, **dataset_loading_kwargs)
    train_dataset = dataset[train_split]
    if add_dataset_indices:
        train_dataset = add_indices(train_dataset)
    if process_line_fn is not None:
        if isinstance(process_line_fn, (list, tuple)):
            processed_train_datasets = []
            for fn in process_line_fn:
                processed = train_dataset.map(
                    lambda x, fn=fn: fn(x, tokenizer))
                processed_train_datasets.append(processed)
            train_dataset = concatenate_datasets(
                processed_train_datasets)
        else:
            print('not loading from cache')
            train_dataset = train_dataset.map(
                lambda x: process_line_fn(x,  tokenizer))
    if val_split is None:
        val_dataset = None
    else:
        val_dataset = dataset[val_split]
        if add_dataset_indices:
            val_dataset = add_indices(val_dataset)
        if process_line_fn is not None:
            if isinstance(process_line_fn, (list, tuple)):
                processed_val_datasets = []
                for fn in process_line_fn:
                    processed = val_dataset.map(
                        lambda x, fn=fn: fn(x, tokenizer))
                    processed_val_datasets.append(processed)
                val_dataset = concatenate_datasets(
                    processed_val_datasets)
            else:
                val_dataset = val_dataset.map(
                    lambda x: process_line_fn(x,  tokenizer))
    if keep_columns is not None:
        train_dataset = train_dataset.remove_columns(
            [col for col in train_dataset.column_names
             if col not in keep_columns])

    if artificial_epochs is not None:
        assert artificial_epochs == 1, (
            'Artificial epoch, moved to GRPO to avoid shuffling samples between'
            ' different epochs.')

        train_dataset = concatenate_datasets(
            [train_dataset]*artificial_epochs)
    out_data = dict(
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    if completion_only_training:
        out_data['data_collator'] = make_masked_sft_collator(
            tokenizer=tokenizer,
            model_name=model_name_or_path,
            custom_start_of_response=custom_start_of_response,
        )
    return out_data
