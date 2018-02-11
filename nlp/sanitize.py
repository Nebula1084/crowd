from nltk.tokenize import RegexpTokenizer


def standardize(df, text_field):
    df = clean(df, text_field)
    tokenizer = RegexpTokenizer(r'\w+')
    df["tokens"] = df[text_field].apply(tokenizer.tokenize)

    return df


def clean(df, text_field):
    df[text_field] = df[text_field].str.replace(r"http\S+", "")
    df[text_field] = df[text_field].str.replace(r"http", "")
    df[text_field] = df[text_field].str.replace(r"@\S+", "")
    df[text_field] = df[text_field].str.replace(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ")
    df[text_field] = df[text_field].str.replace(r"@", "at")
    df[text_field] = df[text_field].str.lower()
    return df
