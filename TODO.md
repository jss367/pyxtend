## TODO
* Needs better performance on cases like np.zeros((10000, 2, 256, 256, 3))
  * Should it only do the extra thing for sets?

If it's unsupported, should it be:
return "unsupported"  # type(obj).__name__ or {type(obj).__name__: 'unsupported'}



Better support for BatchEncoding. Example:
    texts = ["this is my test text", "this is another test text"] * 100

    # load a pre-trained tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(
        "bert-base-uncased",
        add_special_tokens=True,
        max_length=input_dimensions,
        pad_to_max_length=True,
        return_tensors="pt",
    )

    encoded_text = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")


Would be nice if it gave more info on dataframes... number of columns, rows, etc. Maybe something like df.summary()?


Fix this bug:
<img width="401" alt="image" src="https://github.com/jss367/pyxtend/assets/3067731/b86912c0-3283-4bfd-8f9a-14418e0d44ed">

