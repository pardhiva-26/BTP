import os
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from transformers import CLIPTokenizer

# --- Utility: ensure required NLTK packages are available (one-time) ---
# run these lines once in your environment if you haven't downloaded them:
# nltk.download('punkt')
# nltk.download('stopwords')


def generate_corpus3_for_retrieval(data_path):
    """
    Read Corpus3.csv, split Origin Document into sentences, and write
    sentence-level CSV to supplementary/Corpus3_sentence_level.csv
    """
    ORIGIN_LINK_CORPUS = "Corpus3.csv"
    in_path = os.path.join(data_path, ORIGIN_LINK_CORPUS)
    df_evidence = pd.read_csv(in_path, encoding="utf8")

    claim_id_list = []
    relevant_document_id_list = []
    paragraph_id_list = []
    paragraph_list = []

    for _, row in df_evidence.iterrows():
        claim_id = row.get("claim_id")
        relevant_document_id = row.get("relevant_document_id")
        relevant_document = row.get("Origin Document")

        # skip missing documents early
        if pd.isna(relevant_document):
            continue

        # remove simple html tags if any
        if isinstance(relevant_document, str):
            relevant_document = relevant_document.replace("<p>", "").replace("</p>", "")

        # split into sentences
        cur_paragraph_list = sent_tokenize(relevant_document)
        paragraph_id = 0
        for paragraph in cur_paragraph_list:
            if not pd.isna(paragraph) and paragraph != "N/A" and paragraph.strip() != "":
                claim_id_list.append(claim_id)
                relevant_document_id_list.append(relevant_document_id)
                paragraph_list.append(paragraph)
                paragraph_id_list.append(paragraph_id)
                paragraph_id += 1

    out_dir = os.path.join(data_path, "supplementary")
    os.makedirs(out_dir, exist_ok=True)

    df = pd.DataFrame({
        "claim_id": claim_id_list,
        "relevant_document_id": relevant_document_id_list,
        "paragraph_id": paragraph_id_list,
        "paragraph": paragraph_list
    })
    out_file = os.path.join(out_dir, "Corpus3_sentence_level.csv")
    df.to_csv(out_file, index=False)
    print(f"Wrote {out_file} with {len(df)} rows.")


def generate_corpus_id_corpus3_for_retrieval(data_path):
    """
    Read the sentence-level Corpus3 CSV and insert a corpus_id column.
    corpus_id = "<claim_id>-<relevant_document_id>-<paragraph_id>"
    """
    in_file = os.path.join(data_path, "supplementary", "Corpus3_sentence_level.csv")
    df_evidence = pd.read_csv(in_file, encoding="utf8")

    corpus_id_list = []
    for _, row in df_evidence.iterrows():
        claim_id = row["claim_id"]
        relevant_document_id = row["relevant_document_id"]
        paragraph_id = row["paragraph_id"]
        corpus_id = f"{claim_id}-{relevant_document_id}-{paragraph_id}"
        corpus_id_list.append(corpus_id)

    df_evidence.insert(3, "corpus_id", corpus_id_list)
    df_evidence.to_csv(in_file, index=False)
    print(f"Inserted corpus_id and updated {in_file}.")


class Paragraph:
    def __init__(self, corpus_id: str, paragraph_text: str, stopwords_set=None):
        if stopwords_set is None:
            stopwords_set = set(stopwords.words('english'))
        # tokenization and normalization: lowercase and remove stopwords
        tokens = [t.lower() for t in word_tokenize(paragraph_text)]
        # keep tokens that are not stopwords and not pure punctuation
        self.token_set = {t for t in tokens if t not in stopwords_set and any(ch.isalnum() for ch in t)}
        self.corpus_id = corpus_id
        self.paragraph_text = paragraph_text
        self.token_set_len = len(self.token_set)


def load_corpus3_for_retrieval():
    """
    Load supplementary/Corpus3_sentence_level.csv and return a dict:
    { claim_id: { corpus_id: Paragraph(...) , ... }, ... }
    """
    corpus_file = os.path.join("/scratch/sg/nagendra/Mocheg/data", "supplementary", "Corpus3_sentence_level.csv")
    df_corpus = pd.read_csv(corpus_file, encoding="utf8")
    corpus_dic = {}
    sw_set = set(stopwords.words('english'))

    for _, row in df_corpus.iterrows():
        claim_id = row["claim_id"]
        corpus_id = f"{claim_id}-{row['relevant_document_id']}-{row['paragraph_id']}"
        paragraph = row["paragraph"]

        p = Paragraph(corpus_id, paragraph, stopwords_set=sw_set)

        if claim_id not in corpus_dic:
            corpus_dic[claim_id] = {corpus_id: p}
        else:
            corpus_dic[claim_id][corpus_id] = p

    return corpus_dic


def record_negative_corpus_id(relevant_dic, max_neg=100):
    """
    Return up to max_neg corpus_ids from relevant_dic as negative candidates.
    Keeps order stable by sorting keys (if keys are strings, sorts lexicographically).
    """
    keys = list(relevant_dic.keys())
    length = min(max_neg, len(keys))
    return keys[:length]


def add_negative_corpus_id(negative_corpus_id_list, rel_docs_list, claim_id):
    for negative_corpus_id in negative_corpus_id_list:
        rel_docs_list.append([claim_id, 0, negative_corpus_id, 0])
    return rel_docs_list


def similarity(X: str, Y_token_set: set, Y_token_set_len: int, stopwords_set=None) -> float:
    """
    Simple set-based cosine-like similarity (as in original).
    Returns a float in [0,1].
    """
    if stopwords_set is None:
        stopwords_set = set(stopwords.words('english'))

    X_tokens = [t.lower() for t in word_tokenize(X)]
    X_set = {t for t in X_tokens if t not in stopwords_set and any(ch.isalnum() for ch in t)}

    if len(X_set) == 0 and Y_token_set_len == 0:
        return 0.0

    inter_len = len(X_set & Y_token_set)
    # cosine-like normalization used originally:
    denom = 0.5 * (len(X_set) + float(Y_token_set_len))
    if denom == 0:
        return 0.0
    cosine = inter_len / denom
    return cosine


def isin(paragraph_obj: Paragraph, evidence_sent_list: list, threshold: float = 0.85) -> bool:
    """
    Check whether any evidence sentence is similar enough to paragraph_obj.
    """
    for evidence_sent in evidence_sent_list:
        if not pd.isna(evidence_sent) and isinstance(evidence_sent, str) and evidence_sent.strip() != "":
            if similarity(evidence_sent, paragraph_obj.token_set, paragraph_obj.token_set_len) > threshold:
                return True
    return False


def save(rel_docs_list, data_path):
    df = pd.DataFrame(rel_docs_list, columns=['TOPIC', 'ITERATION', 'DOCUMENT#', 'RELEVANCY'])
    out_file = os.path.join(data_path, "text_evidence_qrels_sentence_level.csv")
    df.to_csv(out_file, index=False)  # qrels.csv
    from datetime import datetime
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print(f"Saved {len(df)} rows to {out_file}. Current Time = {current_time}")


def generate_rel(data_path, query_filename="Corpus2.csv"):
    """
    For each claim in query_filename, find matching sentence-level corpus entries from the corpus index.
    Produces text_evidence_qrels_sentence_level.csv in data_path.
    """
    corpus_dic = load_corpus3_for_retrieval()  # {claim_id: {corpus_id: Paragraph}}
    rel_docs_list = []

    query_file = os.path.join(data_path, query_filename)
    df_evidence = pd.read_csv(query_file, encoding="utf8")

    cur_claim_id = None
    # We'll compute negative list when we encounter a claim that exists in corpus_dic
    negative_corpus_id_list = []

    total = len(df_evidence)
    for idx, row in df_evidence.iterrows():
        claim_id = row["claim_id"]
        evidence = row.get("Evidence")
        if idx % 100 == 0:
            print(f"{idx}/{total}")
            # Save intermediate results
            save(rel_docs_list, data_path)

        if claim_id in corpus_dic.keys():
            relevant_dic = corpus_dic[claim_id]  # defined before use
            # update negative list only when claim changes
            if claim_id != cur_claim_id:
                negative_corpus_id_list = record_negative_corpus_id(relevant_dic)
                cur_claim_id = claim_id

            if not pd.isna(evidence) and isinstance(evidence, str) and evidence.strip() != "":
                evidence = evidence.replace("<p>", "").replace("</p>", "")
                evidence_sent_list = sent_tokenize(evidence)
                has_found = False
                for corpus_id, paragraph in relevant_dic.items():
                    if isin(paragraph, evidence_sent_list):
                        has_found = True
                        rel_docs_list.append([claim_id, 0, corpus_id, 1])
                if has_found:
                    # add negatives for this claim once when found
                    rel_docs_list = add_negative_corpus_id(negative_corpus_id_list, rel_docs_list, claim_id)

    # final save
    save(rel_docs_list, data_path)


def truncate_text(text: str, tokenizer_for_truncation: CLIPTokenizer) -> str:
    """
    Tokenize+truncate a single text and decode back to string without special tokens.
    Uses tokenizer.encode / tokenizer.decode for clarity.
    """
    # some tokenizers accept encode directly
    input_ids = tokenizer_for_truncation.encode(text, truncation=True)  # list of ints
    decoded_text = tokenizer_for_truncation.decode(input_ids, skip_special_tokens=True)
    return decoded_text


def preprocess_truncate_claim(data_path, query_filename="Corpus2.csv", out_filename="Corpus2_for_retrieval.csv"):
    """
    Truncate claims in query file using CLIP tokenizer (to model max len) and write new file.
    """
    query_file = os.path.join(data_path, query_filename)
    df_evidence = pd.read_csv(query_file, encoding="utf8")

    tokenizer_for_truncation = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    total = len(df_evidence)
    for idx, row in df_evidence.iterrows():
        claim = row.get("Claim")
        if idx % 100 == 0:
            print(f"{idx}/{total}")
        if not pd.isna(claim) and isinstance(claim, str) and claim.strip() != "":
            query = truncate_text(claim, tokenizer_for_truncation)
            df_evidence.loc[idx, 'Claim'] = query

    out_file = os.path.join(data_path, out_filename)
    df_evidence.to_csv(out_file, index=False)
    print(f"Wrote truncated claims to {out_file}")


if __name__ == "__main__":
    # Example usage:
    # set your base data dir here (replace path as needed)
    base_data_dir = "/scratch/sg/nagendra/Mocheg/data"

    # # 1) generate sentence-level corpus (only needed once if Corpus3.csv changed)
    # generate_corpus3_for_retrieval(base_data_dir)

    # # 2) add corpus_id column to the generated file
    # generate_corpus_id_corpus3_for_retrieval(base_data_dir)

    # # 3) generate retrieval qrels for train/test/val (if Corpus2.csv present in each folder)
    # # if you have separate splits, call generate_rel for each split directory
    # generate_rel(os.path.join(base_data_dir, "test"), query_filename="Corpus2.csv")
    # generate_rel(os.path.join(base_data_dir, "train"), query_filename="Corpus2.csv")
    # generate_rel(os.path.join(base_data_dir, "val"), query_filename="Corpus2.csv")

    # 4) preprocess claims (truncate) and write out
    preprocess_truncate_claim(os.path.join(base_data_dir, "train"), query_filename="Corpus2.csv", out_filename="Corpus2_for_retrieval.csv")
    preprocess_truncate_claim(os.path.join(base_data_dir, "test"), query_filename="Corpus2.csv", out_filename="Corpus2_for_retrieval.csv")
    preprocess_truncate_claim(os.path.join(base_data_dir, "val"), query_filename="Corpus2.csv", out_filename="Corpus2_for_retrieval.csv")

    print("Done.")
